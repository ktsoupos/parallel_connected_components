#!/bin/bash
#SBATCH --job-name=cc_mpi
#SBATCH --partition=batch
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/mpi_%j.out
#SBATCH --error=logs/mpi_%j.err

# MPI Connected Components Job

echo "Job started at: $(date)"
echo "Running on nodes: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of tasks: $SLURM_NTASKS"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Load MPI module (must match build environment)
module purge
module load gcc
module load openmpi

# Configure MPI/UCX to avoid warnings and use shared memory
export UCX_WARN_UNUSED_ENV_VARS=n
export OMPI_MCA_btl=^openib

# Run MPI benchmarks with different process counts
for procs in 2 4 8; do
    echo "========================================="
    echo "Running with $procs processes"
    echo "========================================="

    echo "MPI Shiloach-Vishkin algorithm:"
    srun -n $procs ./build/cc_mpi data/medium_graph.mtx 10000

done

echo "Job completed at: $(date)"
