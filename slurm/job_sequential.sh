#!/bin/bash
#SBATCH --job-name=cc_sequential
#SBATCH --partition=batch
#SBATCH --time=30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/sequential_%j.out
#SBATCH --error=logs/sequential_%j.err

# Sequential Connected Components Job

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run sequential benchmarks
echo "Running sequential Shiloach-Vishkin..."
./build/cc_sequential data/small_graph.mtx 0

echo "Running sequential Shiloach-Vishkin on medium graph..."
./build/cc_sequential data/medium_graph.mtx 0

echo "Job completed at: $(date)"
