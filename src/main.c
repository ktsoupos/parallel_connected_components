#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include "graph.h"
#include "mtx_reader.h"
#include "cc_sequential.h"

int main(const int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph.mtx> [report_interval]\n", argv[0]);
        fprintf(stderr, "  graph.mtx: Matrix Market format graph file\n");
        fprintf(stderr, "  report_interval: Optional progress report interval (0 = silent)\n");
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];
    int32_t report_interval = 0;

    if (argc >= 3) {
        char* endptr;
        errno = 0;
        const long val = strtol(argv[2], &endptr, 10);

        if (errno != 0 || endptr == argv[2] || *endptr != '\0' || val < 0 || val > INT32_MAX) {
            fprintf(stderr, "Error: Invalid report_interval value '%s'\n", argv[2]);
            return EXIT_FAILURE;
        }

        report_interval = (int32_t)val;
    }

    printf("=== Connected Components - Sequential Version ===\n\n");

    /* Read graph from MTX file */
    Graph* g = read_mtx_file_verbose(filename, report_interval);
    if (g == NULL) {
        fprintf(stderr, "Error: Failed to read graph from '%s'\n", filename);
        return EXIT_FAILURE;
    }

    printf("\n");
    graph_print_stats(g);

    /* Run optimized version */
    printf("\n=== Running Optimized Version ===\n");

    clock_t start = clock();
    CCResult* result = label_propagation_min(g);
    clock_t end = clock();

    if (result == NULL) {
        fprintf(stderr, "Error: Optimized algorithm failed\n");
        graph_destroy(g);
        return EXIT_FAILURE;
    }

    const double elapsed_opt = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Optimized algorithm completed in %.3f seconds\n", elapsed_opt);
    cc_result_print_stats(result, g);

    /* Run simple baseline version */
    printf("\n=== Running Simple Baseline Version ===\n");

    start = clock();
    CCResult* result_simple = label_propagation_min_simple(g);
    end = clock();

    if (result_simple == NULL) {
        fprintf(stderr, "Error: Simple algorithm failed\n");
        cc_result_destroy(result);
        graph_destroy(g);
        return EXIT_FAILURE;
    }

    const double elapsed_simple = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Simple algorithm completed in %.3f seconds\n", elapsed_simple);
    cc_result_print_stats(result_simple, g);

    /* Print comparison */
    printf("\n=== Performance Comparison ===\n");
    printf("Optimized: %.3f seconds (%d iterations)\n", elapsed_opt, result->num_iterations);
    printf("Simple:    %.3f seconds (%d iterations)\n", elapsed_simple, result_simple->num_iterations);
    printf("Speedup:   %.2fx\n", elapsed_simple / elapsed_opt);

    /* Cleanup */
    cc_result_destroy(result_simple);
    cc_result_destroy(result);
    graph_destroy(g);

    return EXIT_SUCCESS;
}
