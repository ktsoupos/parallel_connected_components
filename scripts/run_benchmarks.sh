#!/bin/bash
# Performance benchmarking script for reviewers
# Usage: ./scripts/run_benchmarks.sh <graph.mtx>

set -e

GRAPH="$1"

if [ -z "$GRAPH" ] || [ ! -f "$GRAPH" ]; then
    echo "Usage: $0 <graph.mtx>"
    echo "Example: $0 data/ca-CondMat.mtx"
    exit 1
fi

echo "=========================================="
echo "Parallel Connected Components Benchmarks"
echo "=========================================="
echo "Graph: $GRAPH"
echo "Date: $(date)"
echo "System: $(uname -a)"
echo "CPUs: $(nproc) cores"
echo ""

# Check if executables exist
if [ ! -f build/cc_sequential ]; then
    echo "ERROR: Executables not found. Please run 'make' first."
    exit 1
fi

echo "=========================================="
echo "1. Sequential Baseline"
echo "=========================================="
./build/cc_sequential "$GRAPH" 0

echo ""
echo "=========================================="
echo "2. OpenMP Parallel (with scaling)"
echo "=========================================="
for THREADS in 1 2 4 8 16; do
    if [ $THREADS -le $(nproc) ]; then
        echo ""
        echo "--- OpenMP with $THREADS threads ---"
        OMP_NUM_THREADS=$THREADS ./build/cc_openmp "$GRAPH" 0 $THREADS
    fi
done

echo ""
echo "=========================================="
echo "3. PThreads Work-Stealing (with scaling)"
echo "=========================================="
for THREADS in 1 2 4 8 16; do
    if [ $THREADS -le $(nproc) ]; then
        echo ""
        echo "--- PThreads with $THREADS threads ---"
        ./build/cc_pthreads "$GRAPH" 0 $THREADS
    fi
done

if [ -f cmake-build-opencilk/cc_opencilk ]; then
    echo ""
    echo "=========================================="
    echo "4. OpenCilk (with scaling)"
    echo "=========================================="
    for WORKERS in 1 2 4 8 16; do
        if [ $WORKERS -le $(nproc) ]; then
            echo ""
            echo "--- OpenCilk with $WORKERS workers ---"
            CILK_NWORKERS=$WORKERS ./cmake-build-opencilk/cc_opencilk "$GRAPH" 0
        fi
    done
fi

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
