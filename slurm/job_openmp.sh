#!/bin/bash
#SBATCH --job-name=cc_openmp
#SBATCH --partition=batch
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=logs/openmp_%j.out
#SBATCH --error=logs/openmp_%j.err

# OpenMP Connected Components Job

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run OpenMP benchmarks with different thread counts
for threads in 2 4 8; do
    echo "========================================="
    echo "Running with $threads threads"
    echo "========================================="

    export OMP_NUM_THREADS=$threads

    echo "Shiloach-Vishkin algorithm:"
    ./build/cc_openmp data/medium_graph.mtx 0

    echo ""
    echo "Afforest algorithm:"
    ./build/cc_openmp data/medium_graph.mtx 1

    echo ""
done

echo "Job completed at: $(date)"
