#!/bin/bash

# Comprehensive benchmark comparing MPI vs OpenMP implementations
# Usage: ./benchmark_comparison.sh <graph.mtx>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <graph.mtx> [max_processes]"
    echo "Example: $0 data/test_small.mtx 8"
    exit 1
fi

GRAPH=$1
MAX_PROCS=${2:-8}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================"
echo "  Connected Components - Comprehensive Benchmark"
echo "========================================================"
echo "Graph: $GRAPH"
echo "Max processes/threads: $MAX_PROCS"
echo ""

# Check if binaries exist
if [ ! -f "./build/cc_openmp" ]; then
    echo "Error: cc_openmp not found. Build with 'cmake --build build --target cc_openmp'"
    exit 1
fi

if [ ! -f "./build/cc_mpi" ]; then
    echo "Error: cc_mpi not found. Build with 'cmake --build build --target cc_mpi'"
    exit 1
fi

# Sequential baseline
echo -e "${BLUE}=== Sequential Baseline ===${NC}"
./build/cc_sequential $GRAPH 0 > /tmp/cc_seq.log 2>&1
echo ""

# OpenMP benchmarks
echo -e "${GREEN}=== OpenMP Parallel Benchmarks ===${NC}"
for threads in 2 4 8; do
    if [ $threads -le $MAX_PROCS ]; then
        echo -e "${YELLOW}OpenMP with $threads threads:${NC}"
        OMP_NUM_THREADS=$threads ./build/cc_openmp $GRAPH 0 2>&1 | grep -A2 "Performance Summary" | tail -10
        echo ""
    fi
done

# MPI benchmarks
echo -e "${GREEN}=== MPI Distributed Benchmarks ===${NC}"

# Detect if running on SLURM cluster
if [ -n "$SLURM_JOB_ID" ]; then
    MPI_CMD="srun --ntasks"
    echo "Detected SLURM environment, using srun"
else
    MPI_CMD="mpirun -np"
    echo "Using mpirun"
fi

for procs in 2 4 8; do
    if [ $procs -le $MAX_PROCS ]; then
        echo -e "${YELLOW}MPI with $procs processes:${NC}"
        $MPI_CMD $procs ./build/cc_mpi $GRAPH 0 2>&1 | grep -A5 "MPI Performance Summary" | tail -10
        echo ""
    fi
done

echo "========================================================"
echo "  Benchmark Complete"
echo "========================================================"
echo ""
echo "Summary:"
echo "--------"
echo "Sequential baseline: see above"
echo "Best OpenMP: see results above"
echo "Best MPI: see results above"
echo ""
echo "Key Observations:"
echo "- MPI Shiloach-Vishkin should be more accurate than Afforest"
echo "- OpenMP should be faster for shared-memory systems"
echo "- MPI overhead is significant for small graphs"
echo "- For large graphs, MPI can scale beyond single-node limits"
