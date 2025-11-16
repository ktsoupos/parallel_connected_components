#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include "graph.h"
#include "mtx_reader.h"
#include "benchmark.h"

#ifdef _OPENMP
#include "cc_openmp.h"
#endif

#ifdef __cilk
#include "benchmark_opencilk.h"
#include <cilk/cilk.h>

/**
 * Get number of Cilk workers from environment or return default
 */
static int get_cilk_workers(void) {
    const char* workers_env = getenv("CILK_NWORKERS");
    if (workers_env != NULL) {
        const int workers = atoi(workers_env);
        if (workers > 0) {
            return workers;
        }
    }
    // Default to number of processors if not set
    const long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return (nprocs > 0) ? (int)nprocs : 1;
}
#endif

/**
 * Get number of threads for pthreads from environment or return default
 */
static int get_num_threads(void) {
    const char* threads_env = getenv("NUM_THREADS");
    if (threads_env != NULL) {
        const int threads = atoi(threads_env);
        if (threads > 0) {
            return threads;
        }
    }
    // Default to number of processors if not set
    const long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return (nprocs > 0) ? (int)nprocs : 1;
}

int main(const int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph.mtx> [report_interval]\n", argv[0]);
        fprintf(stderr, "  graph.mtx: Matrix Market format graph file\n");
        fprintf(stderr, "  report_interval: Optional progress report interval (0 = silent)\n");
        return EXIT_FAILURE;
    }

    const char *filename = argv[1];
    int32_t report_interval = 0;

    if (argc >= 3) {
        char *endptr;
        errno = 0;
        const long val = strtol(argv[2], &endptr, 10);

        if (errno != 0 || endptr == argv[2] || *endptr != '\0' || val < 0 || val > INT32_MAX) {
            fprintf(stderr, "Error: Invalid report_interval value '%s'\n", argv[2]);
            return EXIT_FAILURE;
        }

        report_interval = (int32_t) val;
    }

#ifdef __cilk
    printf("=== Connected Components - OpenCilk Parallel Version ===\n\n");
#elif defined(_OPENMP)
    printf("=== Connected Components - OpenMP Parallel Version ===\n\n");
#elif defined(USE_PTHREADS)
    printf("=== Connected Components - Pthreads Work-Stealing Version ===\n\n");
#else
    printf("=== Connected Components - Sequential Version ===\n\n");
#endif

    /* Read graph from MTX file */
    Graph *g = read_mtx_file_verbose(filename, report_interval);
    if (g == NULL) {
        fprintf(stderr, "Error: Failed to read graph from '%s'\n", filename);
        return EXIT_FAILURE;
    }

#ifdef __cilk
    /* Run OpenCilk parallel benchmarks */
    const int num_workers = get_cilk_workers();
    const int result = run_opencilk_benchmarks(g, num_workers);
    if (result != 0) {
        graph_destroy(g);
        return EXIT_FAILURE;
    }
#elif defined(_OPENMP)
    /* Run OpenMP parallel benchmarks with default number of threads */
    const int num_threads = get_omp_threads();
    const int result = run_parallel_benchmarks(g, num_threads);
    if (result != 0) {
        graph_destroy(g);
        return EXIT_FAILURE;
    }
#elif defined(USE_PTHREADS)
    /* Run pthreads work-stealing benchmarks */
    const int num_threads = get_num_threads();
    const int result = run_pthreads_benchmarks(g, num_threads);
    if (result != 0) {
        graph_destroy(g);
        return EXIT_FAILURE;
    }
#else
    /* Run sequential benchmarks */
    const int result = run_sequential_benchmarks(g);
    if (result != 0) {
        graph_destroy(g);
        return EXIT_FAILURE;
    }
#endif

    /* Cleanup */
    graph_destroy(g);

    return EXIT_SUCCESS;
}
