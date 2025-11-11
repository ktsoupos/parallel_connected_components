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

## Current Status

- [Done] Graph data structure (CSR format)
- [Done] Matrix Market file reader
- [Done] Main program with CLI
- [TODO] Sequential CC algorithm
- [TODO] OpenMP parallel version
- [TODO] PThreads parallel version
- [TODO] OpenCilk parallel version
