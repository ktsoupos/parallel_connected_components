#include "benchmark.h"
#include "cc_sequential.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

int run_sequential_benchmarks(const Graph* g) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return -1;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);

    /* Run Union-Find (baseline) */
    printf("\n=== Running Union-Find (Baseline) ===\n");
    clock_t start = clock();
    CCResult* result_uf = union_find_cc(g);
    clock_t end = clock();

    if (result_uf == NULL) {
        fprintf(stderr, "Error: Union-Find algorithm failed\n");
        return -1;
    }

    const double elapsed_uf = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Union-Find completed in %.3f seconds\n", elapsed_uf);
    cc_result_print_stats(result_uf, g);

    /* Run Union-Find with edge reordering */
    printf("\n=== Running Union-Find (Edge Reordering) ===\n");
    start = clock();
    CCResult* result_uf_edge = union_find_cc_edge_reorder(g);
    end = clock();

    if (result_uf_edge == NULL) {
        fprintf(stderr, "Error: Union-Find edge reordering algorithm failed\n");
        cc_result_destroy(result_uf);
        return -1;
    }

    const double elapsed_uf_edge = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Union-Find (edge reorder) completed in %.3f seconds\n", elapsed_uf_edge);
    cc_result_print_stats(result_uf_edge, g);

    /* Run optimized label propagation */
    printf("\n=== Running Label Propagation (Optimized) ===\n");
    start = clock();
    CCResult* result_opt = label_propagation_min(g);
    end = clock();

    if (result_opt == NULL) {
        fprintf(stderr, "Error: Optimized label propagation failed\n");
        cc_result_destroy(result_uf);
        return -1;
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
        return -1;
    }

    const double elapsed_simple = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Simple label propagation completed in %.3f seconds\n", elapsed_simple);
    cc_result_print_stats(result_simple, g);

    /* Verify all algorithms produce identical labels */
    printf("\n=== Correctness Verification ===\n");
    bool labels_match = true;

    for (int32_t i = 0; i < num_vertices; i++) {
        if (result_uf->labels[i] != result_uf_edge->labels[i] ||
            result_uf->labels[i] != result_opt->labels[i] ||
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
    printf("LP Simple (baseline):  %.3f seconds (%d iterations)\n", elapsed_simple, result_simple->num_iterations);
    printf("LP Optimized:          %.3f seconds (%d iterations)\n", elapsed_opt, result_opt->num_iterations);
    printf("UF Baseline:           %.3f seconds (1 pass)\n", elapsed_uf);
    printf("UF Edge Reorder:       %.3f seconds (1 pass)\n", elapsed_uf_edge);

    printf("\nSpeedup vs LP Simple (Baseline):\n");
    printf("  LP Optimized:    %.2fx %s\n", elapsed_simple / elapsed_opt,
           elapsed_opt > elapsed_simple ? "slower" : "faster");
    printf("  UF Baseline:     %.2fx %s\n", elapsed_simple / elapsed_uf,
           elapsed_uf > elapsed_simple ? "slower" : "faster");
    printf("  UF Edge Reorder: %.2fx %s\n", elapsed_simple / elapsed_uf_edge,
           elapsed_uf_edge > elapsed_simple ? "slower" : "faster");

    /* Cleanup */
    cc_result_destroy(result_uf);
    cc_result_destroy(result_uf_edge);
    cc_result_destroy(result_opt);
    cc_result_destroy(result_simple);

    return 0;
}
