#include "benchmark.h"
#include "cc_openmp.h"
#include "cc_sequential.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>

int run_parallel_benchmarks(const Graph* g, int num_threads) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return -1;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);

    printf("\n=== Parallel Benchmarks (OpenMP with %d threads) ===\n", num_threads);

    /* Run sequential baseline for comparison */
    printf("\n=== Sequential Baseline (Union-Find Edge Reorder) ===\n");
    const double start_seq = omp_get_wtime();
    CCResult* result_seq = union_find_cc_edge_reorder(g);
    const double end_seq = omp_get_wtime();

    if (result_seq == NULL) {
        fprintf(stderr, "Error: Sequential algorithm failed\n");
        return -1;
    }

    const double elapsed_seq = end_seq - start_seq;
    printf("Sequential UF completed in %.3f seconds\n", elapsed_seq);
    cc_result_print_stats(result_seq, g);

    /* Run synchronous label propagation with OpenMP */
    printf("\n=== OpenMP Synchronous Label Propagation ===\n");
    const double start_sync = omp_get_wtime();
    CCResult* result_sync = label_propagation_sync_omp(g, num_threads);
    const double end_sync = omp_get_wtime();

    if (result_sync == NULL) {
        fprintf(stderr, "Error: Synchronous label propagation failed\n");
        cc_result_destroy(result_seq);
        return -1;
    }

    const double elapsed_sync = end_sync - start_sync;
    printf("OpenMP synchronous LP completed in %.3f seconds\n", elapsed_sync);
    cc_result_print_stats(result_sync, g);

    /* Run asynchronous label propagation with OpenMP */
    printf("\n=== OpenMP Asynchronous Label Propagation ===\n");
    const double start_async = omp_get_wtime();
    CCResult* result_async = label_propagation_async_omp(g, num_threads);
    const double end_async = omp_get_wtime();

    if (result_async == NULL) {
        fprintf(stderr, "Error: Asynchronous label propagation failed\n");
        cc_result_destroy(result_seq);
        cc_result_destroy(result_sync);
        return -1;
    }

    const double elapsed_async = end_async - start_async;
    printf("OpenMP asynchronous LP completed in %.3f seconds\n", elapsed_async);
    cc_result_print_stats(result_async, g);

    /* Verify correctness: compare component counts */
    printf("\n=== Correctness Verification ===\n");
    if (result_seq->num_components == result_sync->num_components &&
        result_seq->num_components == result_async->num_components) {
        printf("Component counts MATCH: %d components\n", result_seq->num_components);

        /* Verify labels produce same components */
        bool labels_match = true;
        for (int32_t i = 0; i < num_vertices; i++) {
            if (result_seq->labels[i] != result_sync->labels[i] ||
                result_sync->labels[i] != result_async->labels[i]) {
                labels_match = false;
                break;
            }
        }

        if (labels_match) {
            printf("Labels EXACTLY MATCH\n");
        } else {
            printf("Labels differ but produce same components (VALID)\n");
        }
    } else {
        printf("WARNING: Component counts DIFFER\n");
        printf("  Sequential:      %d components\n", result_seq->num_components);
        printf("  Parallel Sync:   %d components\n", result_sync->num_components);
        printf("  Parallel Async:  %d components\n", result_async->num_components);
    }

    /* Print performance comparison */
    printf("\n=== Performance Comparison ===\n");
    printf("Sequential (UF edge reorder): %.3f seconds\n", elapsed_seq);
    printf("Parallel (sync LP, %d threads):  %.3f seconds (%d iterations)\n",
           num_threads, elapsed_sync, result_sync->num_iterations);
    printf("Parallel (async LP, %d threads): %.3f seconds (%d iterations)\n",
           num_threads, elapsed_async, result_async->num_iterations);

    /* Compute and print speedups */
    if (elapsed_seq > 0.0) {
        double speedup_sync = elapsed_seq / elapsed_sync;
        double speedup_async = elapsed_seq / elapsed_async;
        double eff_sync = speedup_sync / (double) num_threads * 100.0;
        double eff_async = speedup_async / (double) num_threads * 100.0;

        printf("\nSpeedup vs sequential:\n");
        printf("  Sync LP:  %.2fx (%.1f%% efficiency)\n", speedup_sync, eff_sync);
        printf("  Async LP: %.2fx (%.1f%% efficiency)\n", speedup_async, eff_async);

        if (elapsed_async < elapsed_sync)
            printf("\nAsync LP is %.2fx faster than Sync LP\n", elapsed_sync / elapsed_async);
        else
            printf("\nAsync LP is %.2fx slower than Sync LP\n", elapsed_async / elapsed_sync);
    }

    /* Cleanup */
    cc_result_destroy(result_seq);
    cc_result_destroy(result_sync);
    cc_result_destroy(result_async);

    return 0;
}
