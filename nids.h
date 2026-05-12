#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ============================================================
// CONFIGURATION
// ============================================================
#define ALPHABET_SIZE     256
#define MAX_STATES        512
#define MAX_PATTERNS       32
#define MAX_PACKET_LEN   1500
#define THREADS_PER_BLOCK 256

// ============================================================
// CUDA ERROR-CHECKING MACRO
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
    int      go[MAX_STATES][ALPHABET_SIZE];
    int      fail[MAX_STATES];
    uint32_t output[MAX_STATES];
    int      num_states;
    int      num_patterns;
};

struct MatchResult {
    uint32_t matched_patterns;
    int      match_count;
    int      first_match_pos;
};

// ============================================================
// FUNCTION DECLARATIONS
// ============================================================

// aho_corasick.cu
void build_aho_corasick(AhoCorasickDFA* ac, const char** patterns, int num_patterns);
MatchResult cpu_scan_one_packet(const AhoCorasickDFA* ac, const char* pkt, int len);
size_t generate_packets(char* packet_buf, int* offsets, int* lengths,
                        int num_packets, const char** patterns, int num_patterns);

// kernels.cu
__global__ void scan_packets_kernel(
    const AhoCorasickDFA* __restrict__ dfa,
    const char*           __restrict__ packets,
    const int*            __restrict__ offsets,
    const int*            __restrict__ lengths,
    MatchResult*                       results,
    int                                num_packets);

__global__ void scan_packets_shared_kernel(
    const AhoCorasickDFA* __restrict__ dfa,
    const char*           __restrict__ packets,
    const int*            __restrict__ offsets,
    const int*            __restrict__ lengths,
    MatchResult*                       results,
    int                                num_packets);

__global__ void scan_packets_texture_kernel(
    cudaTextureObject_t                tex_go,
    const AhoCorasickDFA* __restrict__ dfa,
    const char*           __restrict__ packets,
    const int*            __restrict__ offsets,
    const int*            __restrict__ lengths,
    MatchResult*                       results,
    int                                num_packets);