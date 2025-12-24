# Slurm Job Submission Scripts

This directory contains Slurm job submission scripts for running connected components benchmarks on HPC clusters.

## Available Scripts

### Individual Implementation Scripts

- **job_sequential.sh** - Run sequential baseline implementation
- **job_openmp.sh** - Run OpenMP parallel implementation with multiple thread counts
- **job_mpi.sh** - Run MPI distributed implementation with multiple process counts
- **job_cilk.sh** - Run OpenCilk parallel implementation with multiple worker counts
- **job_gpu.sh** - Template for future GPU implementation

### Batch Scripts

- **job_full_benchmark.sh** - Comprehensive benchmark suite running all implementations
- **submit_all.sh** - Convenience script to submit all jobs at once

## Usage

### Submit Individual Jobs

```bash
# Submit a single job
sbatch slurm/job_openmp.sh

# Submit with custom parameters
sbatch --cpus-per-task=16 --time=2:00:00 slurm/job_openmp.sh
```

### Submit All Jobs

```bash
# Submit all benchmark jobs
bash slurm/submit_all.sh
```

### Monitor Jobs

```bash
# View queue
squeue -u $USER

# View job status
squeue -j <jobid>

# View completed job info
sacct -j <jobid>

# View job efficiency
seff <jobid>

# Cancel a job
scancel <jobid>
```

### View Results

```bash
# Follow job output in real-time
tail -f logs/openmp_<jobid>.out

# View completed job output
cat logs/openmp_<jobid>.out
```

## Script Customization

### Common SBATCH Parameters

- `--partition` - Queue to submit to (batch, gpu, testing, etc.)
- `--time` - Maximum runtime (format: hours:minutes:seconds)
- `--nodes` - Number of compute nodes
- `--ntasks` - Total number of MPI tasks
- `--cpus-per-task` - CPU cores per task
- `--mem` - Memory per node (e.g., 4G, 16G)
- `--gres=gpu:N` - Number of GPUs (for GPU partition)

### Example Modifications

Change thread count for OpenMP:
```bash
#SBATCH --cpus-per-task=16
```

Request more memory:
```bash
#SBATCH --mem=32G
```

Use multiple nodes for MPI:
```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
```

## Output Files

- `logs/` - Job output and error files
- `results/job_<jobid>/` - Benchmark results from full benchmark suite

## Resource Recommendations

### Small Graphs (<100K vertices)
- Sequential: 1 CPU, 2G RAM, 5 minutes
- OpenMP: 2-4 CPUs, 4G RAM, 10 minutes
- MPI: 2 processes, 4G RAM, 10 minutes

### Medium Graphs (100K-1M vertices)
- Sequential: 1 CPU, 4G RAM, 15 minutes
- OpenMP: 4-8 CPUs, 8G RAM, 30 minutes
- MPI: 4-8 processes, 8G RAM, 30 minutes

### Large Graphs (>1M vertices)
- Sequential: 1 CPU, 8G RAM, 1 hour
- OpenMP: 8-16 CPUs, 16G RAM, 1 hour
- MPI: 8-16 processes, 16G RAM, 1 hour

## Notes

- Ensure all executables are built before submitting jobs
- Create `logs/` directory before submitting: `mkdir -p logs`
- Adjust partition names based on your cluster configuration
- Module loading commands may need adjustment for your system
