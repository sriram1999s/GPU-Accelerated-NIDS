/*
 ============================================================
  ASSIGNMENT: GPU-Accelerated NIDS Pattern Matching
 ============================================================

  BACKGROUND:
    Network Intrusion Detection Systems (NIDS) like Snort must
    scan every network packet against hundreds of attack signatures.
    At 10 Gbps+, doing this on a CPU is a bottleneck.

    GPUs can process thousands of packets SIMULTANEOUSLY —
    one thread per packet — making them ideal for this task.

  ALGORITHM: Aho-Corasick Multi-Pattern Matching
    Instead of scanning each packet once per pattern (slow),
    we compile ALL patterns into one DFA (Deterministic Finite
    Automaton) on the CPU. Then every GPU thread independently
    walks one packet through that DFA in a single pass.

  YOUR TASKS (marked with TODO 1–5 below):
    TODO 1 — Insert patterns into a trie (goto function)
    TODO 2 — Build failure links to complete the DFA (BFS)
    TODO 3 — Write the GPU kernel DFA simulation loop
    TODO 4 — Calculate grid size and launch the kernel
    TODO 5 — Load output[] into shared memory (optimization)

  BUILD:
    make
    ./gpu_nids

 ============================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <queue>
#include <cuda_runtime.h>

// ============================================================
// CONFIGURATION
// ============================================================
#define ALPHABET_SIZE     256   // Full byte range (0–255)
#define MAX_STATES        512   // Max DFA states
#define MAX_PATTERNS       32   // Must fit in a uint32_t bitmask
#define MAX_PACKET_LEN   1500   // Ethernet MTU in bytes
#define THREADS_PER_BLOCK 256   // Threads per CUDA block (keep as multiple of 32)

// ============================================================
// CUDA ERROR-CHECKING MACRO
//
// Wrap EVERY CUDA API call in this.
// GPU errors are silent without it — this is non-negotiable habit.
// ============================================================
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                         \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d → %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// DATA STRUCTURES
// ============================================================

/*
 * The Aho-Corasick DFA.
 *
 * go[state][char]  → next state  (the transition table)
 * fail[state]      → fallback state when no transition exists
 * output[state]    → bitmask: bit i is set if pattern i matches here
 *
 * After construction, go[][] will have NO -1 entries.
 * Every (state, char) pair has a valid next state.
 * GPU threads can always do:  state = go[state][char]
 * with no special cases.
 *
 * Memory size of go[][]:  512 × 256 × 4 bytes = 512 KB
 */
struct AhoCorasickDFA {
    int      go[MAX_STATES][ALPHABET_SIZE]; // Transition table
    int      fail[MAX_STATES];              // Failure links
    uint32_t output[MAX_STATES];            // Match bitmask per state
    int      num_states;
    int      num_patterns;
};

/* Result written by each GPU thread after scanning its packet */
struct MatchResult {
    uint32_t matched_patterns; // Bitmask of which patterns matched
    int      match_count;      // Number of distinct patterns matched
    int      first_match_pos;  // Byte offset of first match (-1 = none)
};

// ============================================================
// CPU: PHASE 1 — BUILD THE GOTO TRIE
// ============================================================
/*
 * Insert each pattern into a character trie.
 * States are numbered integers; state 0 is the root.
 *
 * Example — patterns "SEL" and "SCRIPT":
 *
 *   root (0)
 *    ├─ 'S' → state 1
 *    │         ├─ 'E' → state 2
 *    │         │         └─ 'L' → state 3  ← output: pattern 0
 *    │         └─ 'C' → state 4
 *    │                   └─ 'R' → state 5
 *    │                             └─ 'I' → state 6
 *    │                                       └─ 'P' → state 7
 *    │                                                 └─ 'T' → state 8 ← output: pattern 1
 *    └─ (other chars): go[0][c] = -1  (filled in Phase 2)
 *
 * go[state][c] = -1 means "no transition yet" — Phase 2 will fill these.
 */
static void build_goto_trie(AhoCorasickDFA* ac,
                             const char**    patterns,
                             int             num_patterns)
{
    // Initialize everything to "no transition" and "no match"
    memset(ac->go,     -1, sizeof(ac->go));
    memset(ac->output,  0, sizeof(ac->output));
    memset(ac->fail,    0, sizeof(ac->fail));
    ac->num_states   = 1;            // State 0 = root (already exists)
    ac->num_patterns = num_patterns;

    for (int p = 0; p < num_patterns; p++) {
        int state = 0; // Start inserting from root

        for (int i = 0; patterns[p][i] != '\0'; i++) {
            int c = (unsigned char)patterns[p][i];

            /* ── TODO 1a ──*/

            if (ac->go[state][c] == -1) {
                ac->go[state][c] = ac->num_states++;

                if (ac->num_states >= MAX_STATES) {
                    fprintf(stderr, "ERROR: MAX_STATES (%d) exceeded. Increase it.\n", MAX_STATES);
                    exit(1);
                }
            }
            state = ac->go[state][c];
        }

        /* ── TODO 1b ── */
        ac->output[state] |= (1u << p);
    }
}

// ============================================================
// CPU: PHASE 2 — BUILD FAILURE LINKS + COMPLETE THE DFA
// ============================================================
/*
 * Failure links are the secret sauce of Aho-Corasick.
 *
 * fail[state] points to the longest PROPER SUFFIX of the string
 * that led to `state` which is also a valid PREFIX of some pattern.
 *
 * Example: if we're at state for "SELEC" and next char doesn't match,
 * fail[] might point us back to the state for "C" (start of another
 * pattern). We never have to restart from scratch.
 *
 * After this function:
 *   - go[][] has NO -1 entries (every missing transition is redirected)
 *   - output[] has been "merged" so states inherit matches from
 *     their failure chain (a state for "XSCRIPT" also reports "SCRIPT")
 *
 * We build failure links using BFS (breadth-first), level by level,
 * so when we process state R, fail[R] is already fully resolved.
 *
 * BFS rules:
 *   Root's children → fail = root
 *   For any other state S reached via edge (R, c):
 *     fail[S] = go[fail[R]][c]
 *                  │
 *                  └─ already valid because we're in BFS order
 *
 * Completing the DFA:
 *   If go[R][c] == -1 (missing), set go[R][c] = go[fail[R]][c]
 *   This "redirects" missing transitions through the failure link.
 */
static void build_failure_and_complete(AhoCorasickDFA* ac)
{
    std::queue<int> q;

    /* ── TODO 2a — Handle root's direct children ── */

    for (int c = 0; c < 256; c++) {
        if(ac->go[0][c] == -1) {
            ac->go[0][c] = 0;
        }
        else {
            ac->fail[ac->go[0][c]] = 0;
            q.push(ac->go[0][c]);
        }
    }

    /* ── TODO 2b — BFS: process every other state ── */

    while (!q.empty()) {
        auto r = q.front();
        q.pop();

        // output merging
        ac->output[r] |= ac->output[ac->fail[r]];

        for (int c = 0; c < 256; c++) {
            if(ac->go[r][c] == -1) {
                ac->go[r][c] = ac->go[ac->fail[r]][c];
            }
            else {
                int S = ac->go[r][c];
                ac->fail[S] = ac->go[ac->fail[r]][c];
                q.push(S);
            }
        } 
    }
}

/* Public builder — calls both phases */
void build_aho_corasick(AhoCorasickDFA* ac,
                         const char**    patterns,
                         int             num_patterns)
{
    build_goto_trie(ac, patterns, num_patterns);
    build_failure_and_complete(ac);
    printf("[CPU] DFA built: %d patterns → %d states\n",
           num_patterns, ac->num_states);
}

// ============================================================
// GPU KERNEL v1 — Basic (global memory)
// ============================================================
/*
 * Each GPU thread scans ONE packet through the DFA.
 *
 * CUDA THREAD INDEXING:
 *
 *   The GPU launches a "grid" of "blocks", each containing "threads".
 *
 *   Grid
 *   └── Block 0 (256 threads)    Block 1 (256 threads)   ...
 *       ├── Thread 0              ├── Thread 0
 *       ├── Thread 1              ├── Thread 1
 *       └── ...                  └── ...
 *
 *   Global thread ID:
 *     tid = blockIdx.x * blockDim.x + threadIdx.x
 *
 *   Thread tid scans packet tid.
 *
 * PARAMETERS:
 *   dfa      — the automaton (read-only, in GPU global memory)
 *   packets  — all packets packed into one big byte array
 *   offsets  — offsets[i] = start byte of packet i in `packets`
 *   lengths  — lengths[i] = byte count of packet i
 *   results  — output array, one MatchResult per packet
 */
__global__ void scan_packets_kernel(
    const AhoCorasickDFA* __restrict__ dfa,
    const char*           __restrict__ packets,
    const int*            __restrict__ offsets,
    const int*            __restrict__ lengths,
    MatchResult*                       results,
    int                                num_packets)
{
    /* ── Compute this thread's global index ── */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* ── Guard: we may launch more threads than packets ── */
    if (tid >= num_packets) return;

    /* ── Locate this thread's packet ── */
    const char* pkt = packets + offsets[tid];
    int         len = lengths[tid];

    /* ── Simulate the DFA ── */
    int      state       = 0;    // Start at root state
    uint32_t matched     = 0;    // Accumulated match bitmask
    int      match_count = 0;
    int      first_pos   = -1;

    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)pkt[i];

        /* ── TODO 3a ── */
        
        state = dfa->go[state][c];

        /* ── TODO 3b ── */
        if (dfa->output[state] != 0) {
            if (first_pos == -1) first_pos = i;
            uint32_t new_matches = dfa->output[state] & ~matched;
            match_count += __popc(new_matches);
            matched |= new_matches;
        } 
    }

    /* ── Write this thread's results to global memory ── */
    results[tid].matched_patterns = matched;
    results[tid].match_count      = match_count;
    results[tid].first_match_pos  = first_pos;
}

// ============================================================
// GPU KERNEL v2 — Shared Memory Optimization
// ============================================================
/*
 * SHARED MEMORY vs GLOBAL MEMORY:
 *
 *   Global memory (DRAM):   ~200-400 cycle latency, large (GBs)
 *   Shared memory (on-chip): ~5 cycle latency,  small (48–164 KB per SM)
 *
 * The output[] array is 512 × 4 = 2 KB — it fits in shared memory.
 * All 256 threads in a block share it, so we only pay the global
 * memory cost ONCE per block instead of once per thread.
 *
 * Pattern:
 *   1. Declare  __shared__ uint32_t s_output[MAX_STATES];
 *   2. All threads cooperatively load output[] into s_output[]
 *   3. Call __syncthreads() — wait for ALL threads to finish loading
 *   4. Replace dfa->output[state]  with  s_output[state]  in the loop
 */
__global__ void scan_packets_shared_kernel(
    const AhoCorasickDFA* __restrict__ dfa,
    const char*           __restrict__ packets,
    const int*            __restrict__ offsets,
    const int*            __restrict__ lengths,
    MatchResult*                       results,
    int                                num_packets)
{
    /* ── TODO 5a — Declare shared memory ────────────────────────
     * __shared__ uint32_t s_output[MAX_STATES];
     * ────────────────────────────────────────────────────────── */

    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;   // This thread's index within its block (0..blockDim.x-1)

    /* ── TODO 5b — Cooperatively load output[] into shared memory
     *
     * We have blockDim.x threads but MAX_STATES elements to load.
     * Use a strided loop so every element gets loaded:
     *
     *   for (int i = lane; i < dfa->num_states; i += blockDim.x)
     *       s_output[i] = dfa->output[i];
     *
     * Then call __syncthreads() to ensure ALL threads have finished
     * loading before anyone starts reading s_output[].
     * ────────────────────────────────────────────────────────── */

    if (tid >= num_packets) return;

    const char* pkt = packets + offsets[tid];
    int         len = lengths[tid];

    int      state       = 0;
    uint32_t matched     = 0;
    int      match_count = 0;
    int      first_pos   = -1;

    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)pkt[i];
        state = dfa->go[state][c];

        /* ── TODO 5c ─────────────────────────────────────────────
         * Same match-check logic as TODO 3b, BUT:
         * Read from s_output[state] instead of dfa->output[state].
         * This is the payoff for all the shared memory setup above.
         * ────────────────────────────────────────────────────── */
    }

    results[tid].matched_patterns = matched;
    results[tid].match_count      = match_count;
    results[tid].first_match_pos  = first_pos;
}

// ============================================================
// CPU: REFERENCE SCANNER (already implemented — use to verify GPU)
// ============================================================
/*
 * Runs the same DFA logic on the CPU.
 * After your GPU kernel is working, compare its output to this.
 * Any mismatch means a bug in your kernel or DFA construction.
 */
static MatchResult cpu_scan_one_packet(const AhoCorasickDFA* ac,
                                        const char*           pkt,
                                        int                   len)
{
    MatchResult result = {0, 0, -1};
    int state = 0;
    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)pkt[i];
        state = ac->go[state][c];
        uint32_t out = ac->output[state];
        if (out) {
            if (result.first_match_pos == -1) result.first_match_pos = i;
            uint32_t new_m = out & ~result.matched_patterns;
            result.match_count += __builtin_popcount(new_m);
            result.matched_patterns |= out;
        }
    }
    return result;
}

// ============================================================
// CPU: SYNTHETIC PACKET GENERATOR (already implemented)
// ============================================================
/*
 * Fills packet_buf with num_packets random payloads.
 * ~10% of packets have a pattern injected at a random offset.
 * Returns total bytes written.
 */
static size_t generate_packets(char*        packet_buf,
                                int*         offsets,
                                int*         lengths,
                                int          num_packets,
                                const char** patterns,
                                int          num_patterns)
{
    srand(12345);
    size_t cursor = 0;
    for (int i = 0; i < num_packets; i++) {
        offsets[i] = (int)cursor;
        int len = 64 + rand() % (MAX_PACKET_LEN - 63);
        lengths[i] = len;
        for (int j = 0; j < len; j++)
            packet_buf[cursor + j] = (char)(32 + rand() % 95);
        if (rand() % 10 == 0 && num_patterns > 0) {
            int p       = rand() % num_patterns;
            int pat_len = (int)strlen(patterns[p]);
            if (pat_len < len) {
                int pos = rand() % (len - pat_len);
                memcpy(packet_buf + cursor + pos, patterns[p], pat_len);
            }
        }
        cursor += len;
    }
    return cursor;
}

// ============================================================
// MAIN
// ============================================================
int main(void)
{
    printf("=========================================\n");
    printf("  GPU-Accelerated NIDS Pattern Matcher   \n");
    printf("=========================================\n\n");

    // ── 1. Patterns (Snort-style attack signatures) ─────────────
    const char* patterns[] = {
        "User-Agent: Nmap",
        "SELECT * FROM",
        "/etc/passwd",
        "cmd.exe",
        "<script>",
        "UNION SELECT",
        "../../../",
        "eval(base64_decode",
    };
    int num_patterns = (int)(sizeof(patterns) / sizeof(patterns[0]));

    // ── 2. Build DFA on CPU ──────────────────────────────────────
    AhoCorasickDFA* h_dfa = (AhoCorasickDFA*)malloc(sizeof(AhoCorasickDFA));
    build_aho_corasick(h_dfa, patterns, num_patterns);

    // ── 3. Generate test packets ─────────────────────────────────
    int    num_packets   = 100000;
    size_t buf_capacity  = (size_t)num_packets * MAX_PACKET_LEN;

    char* h_packets = (char*)malloc(buf_capacity);
    int*  h_offsets = (int* )malloc(num_packets * sizeof(int));
    int*  h_lengths = (int* )malloc(num_packets * sizeof(int));

    printf("[CPU] Generating %d packets...\n", num_packets);
    size_t actual_bytes = generate_packets(
        h_packets, h_offsets, h_lengths,
        num_packets, patterns, num_patterns);
    printf("[CPU] Total payload: %.2f MB\n\n", actual_bytes / 1.0e6);

    // ── 4. Allocate GPU memory ───────────────────────────────────
    AhoCorasickDFA* d_dfa;
    char*           d_packets;
    int*            d_offsets;
    int*            d_lengths;
    MatchResult*    d_results;

    CUDA_CHECK(cudaMalloc(&d_dfa,     sizeof(AhoCorasickDFA)));
    CUDA_CHECK(cudaMalloc(&d_packets, actual_bytes));
    CUDA_CHECK(cudaMalloc(&d_offsets, num_packets * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lengths, num_packets * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, num_packets * sizeof(MatchResult)));

    // ── 5. Copy data Host → Device ───────────────────────────────
    CUDA_CHECK(cudaMemcpy(d_dfa,     h_dfa,     sizeof(AhoCorasickDFA),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_packets, h_packets, actual_bytes,              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets, num_packets * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, h_lengths, num_packets * sizeof(int), cudaMemcpyHostToDevice));

    // ── 6. Launch the kernel ─────────────────────────────────────

    /* ── TODO 4a — Calculate grid size ── */
    int grid_size = (num_packets + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("[GPU] Grid: %d blocks × %d threads\n", grid_size, THREADS_PER_BLOCK);

    // Warm-up run (first kernel launch has driver overhead)
    scan_packets_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        d_dfa, d_packets, d_offsets, d_lengths, d_results, num_packets);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── 7. Timed runs ────────────────────────────────────────────
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    /* ── TODO 4b — Time Kernel v1 ── */
    float ms_v1 = 0, ms_v2 = 0;
    // timing code
    CUDA_CHECK(cudaEventRecord(t0));
    scan_packets_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        d_dfa, d_packets, d_offsets, d_lengths, d_results, num_packets);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_v1, t0, t1));
    
    // TEMP !!!
    MatchResult* h_results = (MatchResult*)malloc(num_packets * sizeof(MatchResult));
    CUDA_CHECK(cudaMemcpy(h_results, d_results,
                          num_packets * sizeof(MatchResult),
                          cudaMemcpyDeviceToHost));
    // TEMP !!!
    
    CUDA_CHECK(cudaEventRecord(t0));
    scan_packets_shared_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        d_dfa, d_packets, d_offsets, d_lengths, d_results, num_packets);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_v2, t0, t1));

    // ── 8. Copy results back Device → Host ──────────────────────
    // MatchResult* h_results = (MatchResult*)malloc(num_packets * sizeof(MatchResult));
    // CUDA_CHECK(cudaMemcpy(h_results, d_results,
    //                       num_packets * sizeof(MatchResult),
    //                       cudaMemcpyDeviceToHost));

    // ── 9. Summarize results ─────────────────────────────────────
    int total_alerts = 0;
    int pattern_hits[MAX_PATTERNS] = {0};

    for (int i = 0; i < num_packets; i++) {
        if (h_results[i].matched_patterns) {
            total_alerts++;
            for (int p = 0; p < num_patterns; p++)
                if (h_results[i].matched_patterns & (1u << p))
                    pattern_hits[p]++;
        }
    }

    double gbps_v1 = (actual_bytes * 8.0) / (ms_v1 * 1.0e6);
    double gbps_v2 = (actual_bytes * 8.0) / (ms_v2 * 1.0e6);

    printf("\n── Results ──────────────────────────────────\n");
    printf("  Packets scanned   : %d\n", num_packets);
    printf("  Packets alerted   : %d  (%.1f%%)\n",
           total_alerts, 100.0 * total_alerts / num_packets);
    printf("\n── Performance ──────────────────────────────\n");
    printf("  Kernel v1 (global mem) : %6.2f ms  → %.1f Gbps\n", ms_v1, gbps_v1);
    printf("  Kernel v2 (shared mem) : %6.2f ms  → %.1f Gbps\n", ms_v2, gbps_v2);
    printf("  Shared mem speedup     : %.2fx\n", ms_v1 / ms_v2);

    printf("\n── Per-Signature Hits ───────────────────────\n");
    for (int p = 0; p < num_patterns; p++)
        printf("  [%d] %-28s : %d\n", p, patterns[p], pattern_hits[p]);

    // ── 10. Correctness check ────────────────────────────────────
    printf("\n── Correctness vs CPU (first 2000 packets) ──\n");
    int errors = 0;
    for (int i = 0; i < 2000 && i < num_packets; i++) {
        MatchResult cpu_r = cpu_scan_one_packet(
            h_dfa, h_packets + h_offsets[i], h_lengths[i]);
        if (cpu_r.matched_patterns != h_results[i].matched_patterns) {
            printf("  MISMATCH packet %d: CPU=0x%08X GPU=0x%08X\n",
                   i, cpu_r.matched_patterns, h_results[i].matched_patterns);
            if (++errors > 5) { printf("  (stopped after 5)\n"); break; }
        }
    }
    printf(errors == 0 ? "  All match ✓\n" : "  %d errors\n", errors);

    // ── 11. Cleanup ──────────────────────────────────────────────
    free(h_dfa); free(h_packets); free(h_offsets);
    free(h_lengths); free(h_results);
    cudaFree(d_dfa); cudaFree(d_packets); cudaFree(d_offsets);
    cudaFree(d_lengths); cudaFree(d_results);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    printf("\n[Done]\n");
    return 0;
}
