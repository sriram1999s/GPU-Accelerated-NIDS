#include "nids.h"

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
    /* ── TODO 5a — Declare shared memory ── */

    __shared__ uint32_t s_output[MAX_STATES];

    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x;   // This thread's index within its block (0..blockDim.x-1)

    /* ── TODO 5b — Cooperatively load output[] into shared memory ── */

    for (int i=lane; i < dfa->num_states; i += blockDim.x) {
        s_output[i] = dfa->output[i];
    }

    __syncthreads();

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

        /* ── TODO 5c ── */
        if (s_output[state] != 0) {
            if (first_pos == -1) first_pos = i;
            uint32_t new_matches = s_output[state] & ~matched;
            match_count += __popc(new_matches);
            matched |= new_matches;
        } 
    }

    results[tid].matched_patterns = matched;
    results[tid].match_count      = match_count;
    results[tid].first_match_pos  = first_pos;
}

// ============================================================
// GPU KERNEL v3 — Texture Memory Optimization
// ============================================================
__global__ void scan_packets_texture_kernel(
    cudaTextureObject_t                tex_go,
    const AhoCorasickDFA* __restrict__ dfa,
    const char*           __restrict__ packets,
    const int*            __restrict__ offsets,
    const int*            __restrict__ lengths,
    MatchResult*                       results,
    int                                num_packets)
{
    __shared__ uint32_t s_output[MAX_STATES];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int lane = threadIdx.x;

    for (int i = lane; i < dfa->num_states; i += blockDim.x) {
        s_output[i] = dfa->output[i];
    }

    // barrier synchronization
    __syncthreads();

    if (tid >= num_packets) return;

    const char* pkt = packets + offsets[tid];
    int         len = lengths[tid];

    int      state       = 0;
    uint32_t matched     = 0;
    int      match_count = 0;
    int      first_pos   = -1;

    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)pkt[i];
        state = tex2D<int>(tex_go, c, state);

        /* ── TODO 5c ── */
        if (s_output[state] != 0) {
            if (first_pos == -1) first_pos = i;
            uint32_t new_matches = s_output[state] & ~matched;
            match_count += __popc(new_matches);
            matched |= new_matches;
        } 
    }

    results[tid].matched_patterns = matched;
    results[tid].match_count      = match_count;
    results[tid].first_match_pos  = first_pos;
}
