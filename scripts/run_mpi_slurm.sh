#!/bin/bash
#SBATCH --job-name=mpi_cc          # Job name
#SBATCH --output=results/mpi_%j.out # Standard output (%j = job ID)
#SBATCH --error=results/mpi_%j.err  # Standard error
#SBATCH --nodes=2                   # Number of nodes
#SBATCH --ntasks=8                  # Total number of MPI tasks
#SBATCH --ntasks-per-node=4         # MPI tasks per node
#SBATCH --cpus-per-task=1           # CPU cores per MPI task
#SBATCH --time=02:00:00             # Time limit (HH:MM:SS)
#SBATCH --partition=standard        # Partition name (change as needed)
#SBATCH --mem=16G                   # Memory per node

# Load modules (adjust for your cluster)
# module load gcc/11.2.0
# module load openmpi/4.1.1

# Print job info
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""

# Create results directory
mkdir -p results

# Graph to test (can be passed as argument or set here)
GRAPH=${1:-"data/roadNet-CA.mtx"}

echo "Testing graph: $GRAPH"
echo ""

# Test with different numbers of processes
for NPROCS in 2 4 8; do
    if [ $NPROCS -le $SLURM_NTASKS ]; then
        echo "=========================================="
        echo "Running with $NPROCS MPI processes"
        echo "=========================================="

        srun --ntasks=$NPROCS --unbuffered ./build/cc_mpi $GRAPH 0

        echo ""
        echo ""
    fi
done

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
