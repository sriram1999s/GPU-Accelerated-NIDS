# ============================================================
#  Makefile — GPU Regex / NIDS Pattern Matching
# ============================================================
#
# Usage:
#   make          — build optimised binary
#   make debug    — build with debug symbols + cuda-memcheck support
#   make profile  — build for Nsight/nvprof profiling
#   make clean    — remove build artefacts
#   make run      — build & run
#
# Requirements:
#   CUDA Toolkit ≥ 11.0  (nvcc must be on PATH)
#   GPU compute capability ≥ 5.0  (Maxwell or newer)
# ============================================================

# ── Compiler & flags ────────────────────────────────────────
NVCC       := nvcc
TARGET     := gpu_nids

# -arch=sm_XX : match your GPU's compute capability
#   RTX 30xx / A100 = sm_86, RTX 20xx = sm_75, GTX 10xx = sm_61
# Use -arch=native to auto-detect (CUDA ≥ 11.6)
ARCH       := -arch=sm_75

NVCCFLAGS  := $(ARCH)          \
              -O3               \
              -std=c++14        \
              -Xcompiler -Wall  \
              --use_fast_math

DEBUGFLAGS := $(ARCH)          \
              -g -G             \
              -std=c++14        \
              -Xcompiler -Wall  \
              -DDEBUG

PROFILEFLAGS := $(ARCH)        \
              -O3               \
              -std=c++14        \
              -lineinfo         \
              --use_fast_math

SRC        := gpu_regex.cu

# ── Default target ───────────────────────────────────────────
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<
	@echo "Built: $(TARGET)"

# ── Debug build ──────────────────────────────────────────────
debug: $(SRC)
	$(NVCC) $(DEBUGFLAGS) -o $(TARGET)_debug $<
	@echo "Debug build: $(TARGET)_debug"
	@echo "Run with:  cuda-memcheck ./$(TARGET)_debug"

# ── Profile build ────────────────────────────────────────────
profile: $(SRC)
	$(NVCC) $(PROFILEFLAGS) -o $(TARGET)_profile $<
	@echo "Profile build: $(TARGET)_profile"
	@echo "Run with:  nv-nsight-cu-cli ./$(TARGET)_profile"
	@echo "       or: nvprof ./$(TARGET)_profile"

# ── Run ──────────────────────────────────────────────────────
run: all
	./$(TARGET)

# ── Clean ────────────────────────────────────────────────────
clean:
	rm -f $(TARGET) $(TARGET)_debug $(TARGET)_profile

.PHONY: all debug profile run clean
