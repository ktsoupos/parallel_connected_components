#!/bin/bash
#SBATCH --job-name=cc_benchmark
#SBATCH --partition=batch
#SBATCH --time=2:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# Full Benchmark Suite for Connected Components

echo "========================================="
echo "Connected Components Benchmark Suite"
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create results directory
RESULTS_DIR="results/job_${SLURM_JOB_ID}"
mkdir -p $RESULTS_DIR

# Test graphs
GRAPHS=("data/small_graph.mtx" "data/medium_graph.mtx")

# Run benchmarks for each graph
for graph in "${GRAPHS[@]}"; do
    echo ""
    echo "========================================="
    echo "Benchmarking with graph: $graph"
    echo "========================================="

    # Sequential baseline
    echo "Running sequential baseline..."
    ./build/cc_sequential $graph 0 | tee $RESULTS_DIR/sequential_$(basename $graph).txt

    # OpenMP with different thread counts
    for threads in 2 4 8; do
        echo "Running OpenMP with $threads threads..."
        export OMP_NUM_THREADS=$threads
        ./build/cc_openmp $graph 1 | tee $RESULTS_DIR/openmp_${threads}_$(basename $graph).txt
    done

    # MPI with different process counts
    for procs in 2 4 8; do
        echo "Running MPI with $procs processes..."
        mpirun -np $procs ./build/cc_mpi $graph 1 | tee $RESULTS_DIR/mpi_${procs}_$(basename $graph).txt
    done

    # Cilk with different worker counts
    for workers in 2 4 8; do
        echo "Running Cilk with $workers workers..."
        export CILK_NWORKERS=$workers
        ./build/cc_cilk $graph 1 | tee $RESULTS_DIR/cilk_${workers}_$(basename $graph).txt
    done

    echo ""
done

echo "========================================="
echo "All benchmarks completed!"
echo "Results saved to: $RESULTS_DIR"
echo "Job completed at: $(date)"
echo "========================================="
