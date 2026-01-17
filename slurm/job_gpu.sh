#!/bin/bash
#SBATCH --job-name=cc_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/gpu_%j.out
#SBATCH --error=logs/gpu_%j.err

# GPU Connected Components Job - CUDA Implementation

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Display GPU information
echo ""
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Load CUDA module (adjust module name based on your cluster)
# Uncomment and modify as needed for your cluster
# module load cuda/11.8
# module load cuda/12.0
module load cuda

# Display loaded modules
echo "=== Loaded Modules ==="
module list
echo ""

# Build CUDA version
echo "=== Building CUDA Version ==="
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make cc_cuda
cd ..
echo ""

# Test graphs directory
DATA_DIR="data"

# Run benchmarks on test graphs
echo "=== Running CUDA Benchmarks ==="
echo ""

# Small test graph for correctness
if [ -f "$DATA_DIR/test_small.mtx" ]; then
    echo "Testing on small graph (test_small.mtx):"
    ./build/cc_cuda "$DATA_DIR/test_small.mtx"
    echo ""
fi

# Medium graph - condensed matter collaboration network
if [ -f "$DATA_DIR/ca-CondMat.mtx" ]; then
    echo "Testing on ca-CondMat.mtx:"
    ./build/cc_cuda "$DATA_DIR/ca-CondMat.mtx"
    echo ""
fi

# Large graph - Orkut social network (if available)
if [ -f "$DATA_DIR/com-Orkut.mtx" ]; then
    echo "Testing on com-Orkut.mtx:"
    ./build/cc_cuda "$DATA_DIR/com-Orkut.mtx"
    echo ""
fi

echo "Job completed at: $(date)"
