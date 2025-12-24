#!/bin/bash
#SBATCH --job-name=cc_cilk
#SBATCH --partition=batch
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=logs/cilk_%j.out
#SBATCH --error=logs/cilk_%j.err

# OpenCilk Connected Components Job

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Load OpenCilk if available as a module
# module load opencilk

# Run Cilk benchmarks with different worker counts
for workers in 2 4 8; do
    echo "========================================="
    echo "Running with $workers workers"
    echo "========================================="

    export CILK_NWORKERS=$workers

    echo "Cilk Shiloach-Vishkin algorithm:"
    ./build/cc_cilk data/medium_graph.mtx 0

    echo ""
    echo "Cilk Afforest algorithm:"
    ./build/cc_cilk data/medium_graph.mtx 1

    echo ""
done

echo "Job completed at: $(date)"
