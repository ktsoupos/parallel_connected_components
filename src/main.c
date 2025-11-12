#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "graph.h"
#include "mtx_reader.h"
#include "benchmark.h"

#ifdef _OPENMP
#include "cc_openmp.h"
#endif

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

#ifdef _OPENMP
    printf("=== Connected Components - OpenMP Parallel Version ===\n\n");
#else
    printf("=== Connected Components - Sequential Version ===\n\n");
#endif

    /* Read graph from MTX file */
    Graph *g = read_mtx_file_verbose(filename, report_interval);
    if (g == NULL) {
        fprintf(stderr, "Error: Failed to read graph from '%s'\n", filename);
        return EXIT_FAILURE;
    }

#ifdef _OPENMP
    /* Run parallel benchmarks with default number of threads */
    const int num_threads = get_omp_threads();
    const int result = run_parallel_benchmarks(g, num_threads);
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
