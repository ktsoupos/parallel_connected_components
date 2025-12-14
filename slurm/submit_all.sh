#!/bin/bash

# Submit all jobs and create dependency chain if needed

echo "Submitting Connected Components jobs to Slurm..."

# Create logs directory if it doesn't exist
mkdir -p logs results

# Submit individual jobs
echo "Submitting sequential job..."
SEQ_JOB=$(sbatch slurm/job_sequential.sh | awk '{print $4}')
echo "Sequential job ID: $SEQ_JOB"

echo "Submitting OpenMP job..."
OMP_JOB=$(sbatch slurm/job_openmp.sh | awk '{print $4}')
echo "OpenMP job ID: $OMP_JOB"

echo "Submitting MPI job..."
MPI_JOB=$(sbatch slurm/job_mpi.sh | awk '{print $4}')
echo "MPI job ID: $MPI_JOB"

echo "Submitting Cilk job..."
CILK_JOB=$(sbatch slurm/job_cilk.sh | awk '{print $4}')
echo "Cilk job ID: $CILK_JOB"

echo ""
echo "All jobs submitted!"
echo "Monitor jobs with: squeue -u $USER"
echo "View job details: sacct -j <jobid>"
echo "Check output in: logs/"
