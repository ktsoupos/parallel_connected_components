# Parallel Connected Components

High-performance implementation of connected components algorithms for large undirected graphs using sequential and parallel approaches (OpenMP, OpenCilk, PThreads).

## Dependencies

**Required:** gcc 11.4+, CMake 3.19.2+, PThreads
**Optional:** OpenMP (for `cc_openmp`), OpenCilk (for `cc_opencilk`)

## Quick Start (For Reviewers)

```bash
# Build everything
make

# Run correctness tests
make test

# Run performance benchmarks
make benchmark

# See all options
make help
```

**Full build and test instructions:** See [SUBMISSION.md](SUBMISSION.md)

## Building

### Using Makefile (Recommended)

```bash
make              # Build sequential, OpenMP, and PThreads
make opencilk     # Build OpenCilk (requires OpenCilk compiler)
make clean        # Clean build artifacts
```

### Using CMake Directly

#### Standard Build (Sequential + OpenMP + PThreads)

```bash
# Release (recommended)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Debug
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

**Outputs:** `build/cc_sequential`, `build/cc_openmp`, `build/cc_pthreads`

### OpenCilk Build

```bash
CC=/opt/opencilk/bin/clang cmake -B cmake-build-opencilk -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-opencilk
```

**Outputs:** `cmake-build-opencilk/cc_opencilk`, `cc_opencilk_cilksan` (race detector), `cc_opencilk_cilkscale` (work-span analyzer), `cc_opencilk_cilkscale_bench` (scalability)

### CLion

**Standard:** Open project → Select Debug/Release → Build (Ctrl+F9)

**OpenCilk:** Settings → CMake → Add profile → Set CMake options: `-DCMAKE_C_COMPILER=/opt/opencilk/bin/clang` → Build directory: `cmake-build-opencilk`

## Usage

```bash
./cc_sequential <graph.mtx> [report_interval]
./cc_openmp <graph.mtx> [report_interval] 
./cc_pthreads <graph.mtx> [report_interval]
./cc_opencilk <graph.mtx> [report_interval]
```

**Arguments:**
- `graph.mtx`: Matrix Market format graph file
- `report_interval`: Progress reporting interval (0 = silent, default)

**Example:**
```bash
./build/cc_sequential data/test_small.mtx
OMP_NUM_THREADS=8 ./build/cc_openmp data/ca-CondMat.mtx
CILK_NWORKERS=16 ./cmake-build-opencilk/cc_opencilk data/com-Orkut.mtx
```

## Algorithms

### Sequential
- **Label Propagation** (simple + optimized queue-based)
- **Union-Find** (baseline + edge-reordered)

### OpenMP
- **Label Propagation** (synchronous + asynchronous)
- **Shiloach-Vishkin**
- **Afforest**

### OpenCilk
- **Afforest** (Cilk work-stealing)
- **Recursive Edge-Based**

### PThreads
- **Label Propagation Sync** (barrier-based, static scheduling)
- **Label Propagation Async** (work-stealing scheduler)
- **Afforest Static** (fixed work distribution)
- **Afforest Dynamic** (work-stealing with load balancing)

## Input Format

Matrix Market (`.mtx`) format for undirected graphs. Test graphs included in `data/`:
- `test_small.mtx` - 6 vertices, 5 edges, 2 components
- `ca-CondMat.mtx` - Collaboration network
- `com-Orkut.mtx` - Large social network

## Project Structure

```
src/
├── cc_sequential.c      # Sequential implementations
├── cc_openmp.c          # OpenMP parallel implementations
├── cc_opencilk.c        # OpenCilk parallel implementations
├── pthreads/
│   ├── cc_pthreads.c    # PThreads label propagation
│   ├── cc_afforest.c    # PThreads Afforest
│   └── afforest_simple.c # Simplified Afforest variant
├── graph.c              # CSR graph structure
└── mtx_reader.c         # Matrix Market parser

benchmarks/              # Benchmarking framework
inc/                     # Header files
data/                    # Test graphs
```

## Performance Notes

- Use **Release** builds for benchmarking (`-O3`)
- Control parallelism via `OMP_NUM_THREADS` or `CILK_NWORKERS`
- Afforest typically performs best on large graphs with skewed degree distributions
- Union-Find edge-reordering provides ~4x speedup over simple label propagation on sequential
