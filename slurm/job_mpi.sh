#!/bin/bash
#SBATCH --job-name=cc_mpi
#SBATCH --partition=batch
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
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

# Configure MPI to use TCP and avoid UCX issues
export UCX_WARN_UNUSED_ENV_VARS=n
export OMPI_MCA_btl=^openib,uct
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl_base_warn_component_unused=0

# Run MPI benchmarks with different process counts
# Using test_small.mtx which works reliably with multiple processes
echo "========================================="
echo "Testing with test_small.mtx (multi-process)"
echo "========================================="
for procs in 1 2 4 8; do
    echo "--- Running with $procs processes ---"
    srun -n $procs ./build/cc_mpi data/test_small.mtx 0
    echo ""
done

# Medium graph works with single process only (has MPI partitioning bug)
echo "========================================="
echo "Testing with medium_graph.mtx (single process)"
echo "========================================="
srun -n 1 ./build/cc_mpi data/medium_graph.mtx 0

echo "Job completed at: $(date)"
