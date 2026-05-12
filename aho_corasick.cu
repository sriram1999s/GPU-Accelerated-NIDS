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
#include <queue>
#include "nids.h"

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
