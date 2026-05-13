# GPU-Accelerated NIDS Pattern Matching

A CUDA implementation of the Aho-Corasick multi-pattern matching algorithm for network intrusion detection. Scans 100,000 network packets against 8 attack signatures simultaneously, achieving ~84 Gbps throughput on a single GPU.

## Background

Network Intrusion Detection Systems like Snort must scan every arriving packet against hundreds of attack signatures. At 10 Gbps+, doing this on a CPU becomes a bottleneck. GPUs can process thousands of packets in parallel — one thread per packet — making them a natural fit for this workload.

## How It Works

**Phase 1 — CPU: Build the automaton**

All attack signatures are compiled into a single Aho-Corasick DFA (Deterministic Finite Automaton). Instead of scanning each packet once per pattern, a packet is scanned exactly once regardless of how many patterns exist — O(N) in packet length.

Construction happens in two steps:
- Trie insertion: each pattern is inserted character by character, creating a state for each unique prefix.
- BFS completion: failure links are computed level by level, and all missing transitions are redirected so no `-1` entries remain in the transition table.

**Phase 2 — GPU: Scan packets in parallel**

The completed DFA is uploaded to GPU memory once. Each GPU thread independently scans one packet by walking the DFA transition table byte by byte. With 100,000 packets and 256 threads per block, 391 blocks run concurrently across the GPU's streaming multiprocessors.

Match results are written to a per-thread output slot and copied back to the CPU for reporting.

## Attack Signatures

```
User-Agent: Nmap       — network scanner detection
SELECT * FROM          — SQL injection
/etc/passwd            — Unix path traversal
cmd.exe                — Windows shell invocation
<script>               — XSS attempt
UNION SELECT           — SQL UNION attack
../../../              — directory traversal
eval(base64_decode     — PHP webshell payload
```

## Results (GTX 1650 Ti, sm_75)

```
Packets scanned  : 100,000
Packets alerted  : 9,952  (10%)
DFA states       : 94
Correctness      : All match ✓  (verified against CPU reference)

v1 global mem       : 15.79 ms  →  39.6 Gbps  (baseline)
v2 shared output[]  :  7.43 ms  →  84.2 Gbps  (2.12x v1) ← best
v3 texture go[][]   : 19.24 ms  →  32.5 Gbps  (0.82x v1) ← slower
v4 shared + streams : 14.35 ms  →  43.6 Gbps  (0.52x v2) ← slower
```

## Optimization Notes

**v2 — shared memory for `output[]` (+2.12x)**
The `output[]` array (2 KB) fits in shared memory. All 256 threads in a block
cooperatively load it once, then read it at ~5 cycle latency instead of ~200 cycle
global memory latency. This is the only optimization that meaningfully improved throughput.

**v3 — texture memory for `go[][]` (-0.82x)**
Texture memory is designed for 2D spatially local access. In theory `go[state][c]`
should benefit since threads scanning similar packets walk similar DFA paths. In
practice, the GTX 1650 Ti's texture cache is too small — the 512×256 transition
table (512 KB) thrashes it. Result: slower than the baseline.

**v4 — CUDA streams (-0.52x vs v2)**
Streams overlap H→D memcpy with kernel execution to hide transfer latency. This
benchmark uploads all data upfront before the timed runs, so by the time v4 executes,
the data is already on the GPU — streams add re-upload overhead with no benefit. In a
real NIDS processing a continuous feed of fresh packets, streams would show their advantage.

The real bottleneck throughout is `go[state][c]` — a 512 KB table accessed with a
random pattern (different threads are at different states). This is non-coalesced
global memory access and cannot be fully optimized without restructuring the algorithm.

## Implementation

```
nids.h           — shared structs, macros, defines, function declarations
aho_corasick.cu  — CPU DFA construction (build_goto_trie, build_failure_and_complete)
kernels.cu       — all four GPU kernels
main.cu          — orchestration, timing, correctness check
Makefile         — build system
```

**Key CUDA concepts used:**
- Grid / block / thread indexing
- Global memory allocation and host-device transfers
- Shared memory cooperative loading with `__syncthreads()`
- Texture memory objects (`cudaTextureObject_t`, `tex2D`)
- CUDA streams and `cudaMemcpyAsync`
- Pinned (page-locked) memory with `cudaMallocHost`
- CUDA events for kernel timing
- `__popc()` hardware popcount intrinsic
- Bitmask encoding for multi-pattern match results

## Build

```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Set ARCH in Makefile to match (e.g. sm_86 for RTX 3080)
make

./gpu_nids
```
