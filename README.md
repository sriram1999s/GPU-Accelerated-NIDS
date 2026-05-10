# GPU-Accelerated NIDS Pattern Matching

A CUDA implementation of the Aho-Corasick multi-pattern matching algorithm for network intrusion detection. Scans 100,000 network packets against 8 attack signatures simultaneously, achieving ~40 Gbps throughput on a single GPU.

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

## Results

```
Packets scanned  : 100,000
Packets alerted  : ~9,952  (10%)
DFA states       : 94
Kernel v1        : 15.88 ms  →  39.4 Gbps  (global memory)
Kernel v2        : in progress              (shared memory)
Correctness      : All match ✓  (verified against CPU reference)
```

## Implementation

```
gpu_regex.cu
├── AhoCorasickDFA               — transition table, failure links, output bitmasks
├── build_goto_trie()            — CPU phase 1: insert patterns into trie
├── build_failure_and_complete() — CPU phase 2: BFS failure links + DFA completion
├── scan_packets_kernel          — GPU kernel v1: one thread per packet, global memory
├── scan_packets_shared_kernel   — GPU kernel v2: shared memory output[] cache
├── cpu_scan_one_packet()        — CPU reference implementation for correctness checks
└── main()                       — orchestration, timing, results
```

**Key CUDA concepts used:**
- Grid / block / thread indexing
- Global memory allocation and host-device transfers
- Shared memory cooperative loading with `__syncthreads()`
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