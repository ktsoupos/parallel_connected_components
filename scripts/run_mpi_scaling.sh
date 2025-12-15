#!/bin/bash
#SBATCH --job-name=mpi_cc_scaling   # Job name
#SBATCH --output=results/scaling_%j.out
#SBATCH --error=results/scaling_%j.err
#SBATCH --nodes=4                   # Number of nodes
#SBATCH --ntasks=16                 # Total MPI tasks (adjust based on cluster)
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00             # 4 hour time limit
#SBATCH --partition=standard        # Change to your partition
#SBATCH --mem-per-cpu=4G

# Load required modules (adjust for your cluster)
# module load gcc/11.2.0
# module load openmpi/4.1.1
# module load cmake/3.20

echo "=========================================="
echo "MPI Connected Components Scaling Study"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Create results directory
mkdir -p results

# Define graphs to test
GRAPHS=(
    "data/roadNet-CA.mtx"
    "data/roadNet-PA.mtx"
    "data/europe_osm.mtx"
)

# Process counts to test (powers of 2)
PROCESS_COUNTS=(2 4 8 16)

# Test each graph
for GRAPH in "${GRAPHS[@]}"; do
    if [ ! -f "$GRAPH" ]; then
        echo "Skipping $GRAPH (not found)"
        continue
    fi

    GRAPH_NAME=$(basename "$GRAPH" .mtx)
    echo ""
    echo "=========================================="
    echo "Testing: $GRAPH_NAME"
    echo "=========================================="
    echo ""

    # Test with different process counts
    for NPROCS in "${PROCESS_COUNTS[@]}"; do
        if [ $NPROCS -le $SLURM_NTASKS ]; then
            echo "--------------------------------------"
            echo "Running with $NPROCS processes"
            echo "--------------------------------------"

            # Run benchmark and save to separate file
            OUTPUT_FILE="results/${GRAPH_NAME}_np${NPROCS}_${SLURM_JOB_ID}.txt"

            mpirun -np $NPROCS ./build/cc_mpi $GRAPH 0 | tee "$OUTPUT_FILE"

            echo ""
        fi
    done
done

# Generate summary
echo ""
echo "=========================================="
echo "Generating Summary"
echo "=========================================="

SUMMARY_FILE="results/summary_${SLURM_JOB_ID}.txt"

{
    echo "MPI Connected Components - Scaling Summary"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Date: $(date)"
    echo ""
    echo "=========================================="
    echo ""

    for GRAPH in "${GRAPHS[@]}"; do
        GRAPH_NAME=$(basename "$GRAPH" .mtx)

        if [ -f "results/${GRAPH_NAME}_np2_${SLURM_JOB_ID}.txt" ]; then
            echo "Graph: $GRAPH_NAME"
            echo "--------------------------------------"

            for NPROCS in "${PROCESS_COUNTS[@]}"; do
                FILE="results/${GRAPH_NAME}_np${NPROCS}_${SLURM_JOB_ID}.txt"
                if [ -f "$FILE" ]; then
                    echo ""
                    echo "With $NPROCS processes:"
                    grep -A 6 "MPI Performance Summary" "$FILE" | tail -6
                fi
            done

            echo ""
            echo "=========================================="
            echo ""
        fi
    done
} | tee "$SUMMARY_FILE"

echo ""
echo "Results saved to: results/"
echo "Summary: $SUMMARY_FILE"
echo ""
echo "Job completed at: $(date)"
