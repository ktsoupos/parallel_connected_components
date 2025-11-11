#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
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

    /* Run Union-Find (optimal) */
    printf("\n=== Running Union-Find (Optimal) ===\n");

    clock_t start = clock();
    CCResult* result_uf = union_find_cc(g);
    clock_t end = clock();

    if (result_uf == NULL) {
        fprintf(stderr, "Error: Union-Find algorithm failed\n");
        graph_destroy(g);
        return EXIT_FAILURE;
    }

    const double elapsed_uf = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Union-Find completed in %.3f seconds\n", elapsed_uf);
    cc_result_print_stats(result_uf, g);

    /* Run optimized label propagation */
    printf("\n=== Running Label Propagation (Optimized) ===\n");

    start = clock();
    CCResult* result_opt = label_propagation_min(g);
    end = clock();

    if (result_opt == NULL) {
        fprintf(stderr, "Error: Optimized label propagation failed\n");
        cc_result_destroy(result_uf);
        graph_destroy(g);
        return EXIT_FAILURE;
    }

    const double elapsed_opt = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Optimized label propagation completed in %.3f seconds\n", elapsed_opt);
    cc_result_print_stats(result_opt, g);

    /* Run simple label propagation */
    printf("\n=== Running Label Propagation (Simple) ===\n");

    start = clock();
    CCResult* result_simple = label_propagation_min_simple(g);
    end = clock();

    if (result_simple == NULL) {
        fprintf(stderr, "Error: Simple label propagation failed\n");
        cc_result_destroy(result_uf);
        cc_result_destroy(result_opt);
        graph_destroy(g);
        return EXIT_FAILURE;
    }

    const double elapsed_simple = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Simple label propagation completed in %.3f seconds\n", elapsed_simple);
    cc_result_print_stats(result_simple, g);

    /* Verify all algorithms produce identical labels */
    printf("\n=== Correctness Verification ===\n");
    bool labels_match = true;
    const int32_t num_vertices = graph_get_num_vertices(g);

    for (int32_t i = 0; i < num_vertices; i++) {
        if (result_uf->labels[i] != result_opt->labels[i] ||
            result_uf->labels[i] != result_simple->labels[i]) {
            labels_match = false;
            break;
        }
    }

    if (labels_match) {
        printf("All algorithms produce IDENTICAL labels\n");
    } else {
        printf("WARNING: Algorithms produce DIFFERENT labels\n");
    }

    /* Print comparison */
    printf("\n=== Performance Comparison ===\n");
    printf("Union-Find:    %.3f seconds (1 pass)\n", elapsed_uf);
    printf("LP Optimized:  %.3f seconds (%d iterations)\n", elapsed_opt, result_opt->num_iterations);
    printf("LP Simple:     %.3f seconds (%d iterations)\n", elapsed_simple, result_simple->num_iterations);

    printf("\nSpeedup vs Union-Find:\n");
    printf("  LP Optimized: %.2fx %s\n", elapsed_opt / elapsed_uf,
           elapsed_opt > elapsed_uf ? "slower" : "faster");
    printf("  LP Simple:    %.2fx %s\n", elapsed_simple / elapsed_uf,
           elapsed_simple > elapsed_uf ? "slower" : "faster");
    printf("\nLP Optimization gain: %.2fx\n", elapsed_simple / elapsed_opt);

    /* Cleanup */
    cc_result_destroy(result_uf);
    cc_result_destroy(result_opt);
    cc_result_destroy(result_simple);
    graph_destroy(g);

    return EXIT_SUCCESS;
}
