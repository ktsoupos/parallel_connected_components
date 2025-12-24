# Parallel Connected Components

High-performance implementation of connected components algorithms for large undirected graphs using sequential and parallel approaches (OpenMP, OpenCilk, PThreads, MPI).

## Dependencies

**Required:** gcc 11.4+, CMake 3.19.2+, PThreads
**Optional:** OpenMP (for `cc_openmp`), OpenCilk (for `cc_opencilk`), MPI (for `cc_mpi`)

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
make              # Build sequential, OpenMP, PThreads, and MPI
make opencilk     # Build OpenCilk (requires OpenCilk compiler)
make clean        # Clean build artifacts
```

### Using CMake Directly

#### Standard Build (Sequential + OpenMP + PThreads + MPI)

```bash
# Release (recommended)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Debug
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

**Outputs:** `build/cc_sequential`, `build/cc_openmp`, `build/cc_pthreads`, `build/cc_mpi`

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
mpirun -np <num_processes> ./cc_mpi <graph.mtx> [neighbor_rounds]
```

**Arguments:**
- `graph.mtx`: Matrix Market format graph file
- `report_interval`: Progress reporting interval (0 = silent, default)
- `neighbor_rounds`: (MPI only) Number of neighbor sampling rounds for Afforest (default: 2)
- `num_processes`: (MPI only) Number of MPI processes

**Example:**
```bash
./build/cc_sequential data/test_small.mtx
OMP_NUM_THREADS=8 ./build/cc_openmp data/ca-CondMat.mtx
CILK_NWORKERS=16 ./cmake-build-opencilk/cc_opencilk data/com-Orkut.mtx
mpirun -np 4 ./build/cc_mpi data/ca-CondMat.mtx 2
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

### MPI
- **Afforest** (distributed memory with neighbor sampling and remote edge linking)
- **Shiloach-Vishkin** (hybrid local union-find + distributed hooking)

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
├── mpi/
│   └── cc_mpi.c         # MPI distributed implementations
├── graph.c              # CSR graph structure
└── mtx_reader.c         # Matrix Market parser

benchmarks/              # Benchmarking framework
inc/                     # Header files
data/                    # Test graphs
```

## Performance Notes

- Use **Release** builds for benchmarking (`-O3`)
- Control parallelism via `OMP_NUM_THREADS`, `CILK_NWORKERS`, or `mpirun -np`
- Afforest typically performs best on large graphs with skewed degree distributions
- Union-Find edge-reordering provides ~4x speedup over simple label propagation on sequential
- MPI implementations use distributed memory and are suitable for very large graphs that don't fit in single-node memory
