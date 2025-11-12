# Parallel Connected Components

Implementation of connected components algorithms for undirected graphs using sequential and parallel approaches.

## Overview

This project identifies connected components in large undirected graphs. A connected component is a maximal subgraph where there is a path from each vertex to any other vertex.

## Dependencies

- **gcc** 11.4+ (with C11 support)
- **CMake** 3.10+
- **OpenMP** (included with gcc)
- **PThreads** (standard on Linux)
- **OpenCilk** (optional)

## Building

### Using CMake

```bash
mkdir build
cd build
cmake ..
make
```

### Using CLion

1. Open the project in CLion
2. CLion will automatically detect `CMakeLists.txt`
3. Click **Build → Build Project** (Ctrl+F9)
4. Executables will be in `cmake-build-debug/`

## Running

### Command Line

```bash
./cc_sequential <graph.mtx> [report_interval]
```

**Arguments:**
- `graph.mtx`: Path to Matrix Market format graph file
- `report_interval`: Optional progress reporting (0 = silent)

**Example:**

```bash
./cc_sequential data/test_small.mtx 1000
```

### CLion Configuration

1. **Run → Edit Configurations**
2. Select `cc_sequential`
3. Set **Program arguments**: `data/test_small.mtx 1000`
4. Set **Working directory**: `$ProjectFileDir$`
5. Click **OK** and run (Shift+F10)

## Test Data

A small test graph is included: `data/test_small.mtx`
- 6 vertices
- 5 edges
- 2 connected components

Place additional `.mtx` graph files in the `data/` directory.

## Algorithms

### Sequential Implementations

1. **Label Propagation (Simple)** - Baseline implementation that processes all vertices each iteration
2. **Label Propagation (Optimized)** - Queue-based approach that only processes changed vertices
3. **Union-Find (Baseline)** - Path halving with union-by-minimum
4. **Union-Find (Edge Reorder)** - Optimized to process each undirected edge only once

## Performance

Benchmark on graph with 3,997,962 vertices (single component):

| Algorithm | Time (seconds) | Speedup vs LP Simple |
|-----------|---------------|---------------------|
| LP Simple (baseline) | 0.850 | 1.00x |
| LP Optimized | 0.701 | 1.21x faster |
| UF Baseline | 0.238 | 3.58x faster |
| **UF Edge Reorder** | **0.201** | **4.24x faster** |

**Key Optimizations:**
- **Edge Reordering**: Processes each undirected edge once (u > v check) instead of twice, reducing work by ~50%
- **Path Halving**: Single-pass path compression in union-find
- **Root Caching**: Avoids redundant find operations per vertex
- **Queue-based LP**: Only processes vertices with changed labels

## Current Status

- [Done] Graph data structure (CSR format)
- [Done] Matrix Market file reader
- [Done] Main program with CLI
- [Done] Sequential CC algorithms (4 variants)
- [Done] Benchmarking framework
- [TODO] OpenMP parallel version
- [TODO] PThreads parallel version
- [TODO] OpenCilk parallel version
