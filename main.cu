
#include "nids.h"

// ============================================================
// CPU: REFERENCE SCANNER (already implemented — use to verify GPU)
// ============================================================
MatchResult cpu_scan_one_packet(const AhoCorasickDFA* ac,
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
size_t generate_packets(char*        packet_buf,
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

    // Pinned memory — required for async cudaMemcpyAsync in streams
    char*        h_packets;
    int*         h_offsets;
    int*         h_lengths;
    MatchResult* h_results;
    CUDA_CHECK(cudaMallocHost(&h_packets, buf_capacity));
    CUDA_CHECK(cudaMallocHost(&h_offsets, num_packets * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_lengths, num_packets * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_results, num_packets * sizeof(MatchResult)));

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

    // ── 6. Texture object setup for go[][] ─────────────────────────
    cudaArray_t cu_go_array;
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<int>();
    CUDA_CHECK(cudaMallocArray(&cu_go_array, &channel_desc, ALPHABET_SIZE, MAX_STATES));

    CUDA_CHECK(cudaMemcpy2DToArray(
        cu_go_array, 0, 0,
        h_dfa->go,
        ALPHABET_SIZE * sizeof(int),
        ALPHABET_SIZE * sizeof(int),
        MAX_STATES,
        cudaMemcpyHostToDevice));

    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = cu_go_array;

    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    tex_desc.filterMode       = cudaFilterModePoint;
    tex_desc.readMode         = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    cudaTextureObject_t tex_go = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex_go, &res_desc, &tex_desc, NULL));

    // ── 7. Launch the kernel ─────────────────────────────────────
    int grid_size = (num_packets + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("[GPU] Grid: %d blocks × %d threads\n", grid_size, THREADS_PER_BLOCK);

    // Warm-up run
    scan_packets_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        d_dfa, d_packets, d_offsets, d_lengths, d_results, num_packets);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── 8. Timed runs ────────────────────────────────────────────
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    float ms_v1 = 0, ms_v2 = 0, ms_v3 = 0, ms_v4 = 0;

    // v1 — global memory
    CUDA_CHECK(cudaEventRecord(t0));
    scan_packets_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        d_dfa, d_packets, d_offsets, d_lengths, d_results, num_packets);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_v1, t0, t1));

    // v2 — shared memory
    CUDA_CHECK(cudaEventRecord(t0));
    scan_packets_shared_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        d_dfa, d_packets, d_offsets, d_lengths, d_results, num_packets);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_v2, t0, t1));

    // v3 — texture memory
    CUDA_CHECK(cudaEventRecord(t0));
    scan_packets_texture_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        tex_go, d_dfa, d_packets, d_offsets, d_lengths, d_results, num_packets);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_v3, t0, t1));

    // v4 — shared memory + streams
    int batch = (num_packets + NUM_STREAMS - 1) / NUM_STREAMS;

    CUDA_CHECK(cudaEventRecord(t0));
    for (int s = 0; s < NUM_STREAMS; s++) {
        int start = s * batch;
        int count = min(batch, num_packets - start);
        if (count <= 0) break;

        int grid = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // compute byte range for this batch
        int byte_start = h_offsets[start];
        int byte_end   = h_offsets[start + count - 1] + h_lengths[start + count - 1];
        int byte_count = byte_end - byte_start;

        CUDA_CHECK(cudaMemcpyAsync(d_packets + byte_start,
                                   h_packets + byte_start,
                                   byte_count,
                                   cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_offsets + start,
                                   h_offsets + start,
                                   count * sizeof(int),
                                   cudaMemcpyHostToDevice, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_lengths + start,
                                   h_lengths + start,
                                   count * sizeof(int),
                                   cudaMemcpyHostToDevice, streams[s]));

        scan_packets_shared_kernel<<<grid, THREADS_PER_BLOCK, 0, streams[s]>>>(
            d_dfa, d_packets, d_offsets + start, d_lengths + start,
            d_results + start, count);

        CUDA_CHECK(cudaMemcpyAsync(h_results + start,
                                   d_results + start,
                                   count * sizeof(MatchResult),
                                   cudaMemcpyDeviceToHost, streams[s]));
    }
    for (int s = 0; s < NUM_STREAMS; s++)
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_v4, t0, t1));

    // ── 9. Copy results back Device → Host (for v1/v2/v3 correctness) ──
    CUDA_CHECK(cudaMemcpy(h_results, d_results,
                          num_packets * sizeof(MatchResult),
                          cudaMemcpyDeviceToHost));

    // ── 10. Summarize results ─────────────────────────────────────
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
    double gbps_v3 = (actual_bytes * 8.0) / (ms_v3 * 1.0e6);
    double gbps_v4 = (actual_bytes * 8.0) / (ms_v4 * 1.0e6);

    printf("\n── Results ──────────────────────────────────\n");
    printf("  Packets scanned   : %d\n", num_packets);
    printf("  Packets alerted   : %d  (%.1f%%)\n",
           total_alerts, 100.0 * total_alerts / num_packets);
    printf("\n── Performance ──────────────────────────────\n");
    printf("  v1 global mem          : %6.2f ms  → %6.1f Gbps\n",           ms_v1, gbps_v1);
    printf("  v2 shared output[]     : %6.2f ms  → %6.1f Gbps  (%.2fx v1)\n", ms_v2, gbps_v2, ms_v1/ms_v2);
    printf("  v3 texture go[][]      : %6.2f ms  → %6.1f Gbps  (%.2fx v1)\n", ms_v3, gbps_v3, ms_v1/ms_v3);
    printf("  v4 shared + streams    : %6.2f ms  → %6.1f Gbps  (%.2fx v2)\n", ms_v4, gbps_v4, ms_v2/ms_v4);

    printf("\n── Per-Signature Hits ───────────────────────\n");
    for (int p = 0; p < num_patterns; p++)
        printf("  [%d] %-28s : %d\n", p, patterns[p], pattern_hits[p]);

    // ── 11. Correctness check ────────────────────────────────────
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

    // ── 12. Cleanup ──────────────────────────────────────────────
    for (int s = 0; s < NUM_STREAMS; s++)
        CUDA_CHECK(cudaStreamDestroy(streams[s]));

    free(h_dfa);
    cudaFreeHost(h_packets);
    cudaFreeHost(h_offsets);
    cudaFreeHost(h_lengths);
    cudaFreeHost(h_results);

    cudaFree(d_dfa); cudaFree(d_packets); cudaFree(d_offsets);
    cudaFree(d_lengths); cudaFree(d_results);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaDestroyTextureObject(tex_go);
    cudaFreeArray(cu_go_array);

    printf("\n[Done]\n");
    return 0;
}
