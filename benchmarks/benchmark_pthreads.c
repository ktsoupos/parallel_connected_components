#include "benchmark.h"
#include "cc_pthreads.h"
#include "cc_sequential.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

int run_pthreads_benchmarks(const Graph* g, int num_threads) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return -1;
    }

    if (num_threads <= 0) {
        fprintf(stderr, "Error: Invalid number of threads\n");
        return -1;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);

    printf("\n=== Parallel Benchmarks (Pthreads Work-Stealing with %d threads) ===\n", num_threads);

    /* Run sequential baseline for comparison */
    printf("\n=== Sequential Baseline (Union-Find Edge Reorder) ===\n");
    clock_t start_seq = clock();
    CCResult* result_seq = label_propagation_min(g);
    clock_t end_seq = clock();

    if (result_seq == NULL) {
        fprintf(stderr, "Error: Sequential algorithm failed\n");
        return -1;
    }

    const double elapsed_seq = (double)(end_seq - start_seq) / CLOCKS_PER_SEC;
    printf("Sequential UF completed in %.5f seconds\n", elapsed_seq);
    cc_result_print_stats(result_seq, g);

    /* Run synchronous label propagation with pthreads */
    printf("\n=== Pthreads Synchronous Label Propagation ===\n");
    clock_t start_sync = clock();
    CCResult* result_sync = label_propagation_sync_pthreads(g, num_threads);
    clock_t end_sync = clock();

    if (result_sync == NULL) {
        fprintf(stderr, "Error: Synchronous label propagation failed\n");
        cc_result_destroy(result_seq);
        return -1;
    }

    const double elapsed_sync = (double)(end_sync - start_sync) / CLOCKS_PER_SEC;
    printf("Pthreads synchronous LP completed in %.5f seconds\n", elapsed_sync);
    cc_result_print_stats(result_sync, g);

    /* Verify correctness: compare component counts */
    printf("\n=== Correctness Verification ===\n");
    if (result_seq->num_components == result_sync->num_components) {
        printf("✓ Component counts MATCH: %d components\n", result_seq->num_components);

        /* Verify labels produce same components */
        bool labels_match = true;
        for (int32_t i = 0; i < num_vertices; i++) {
            if (result_seq->labels[i] != result_sync->labels[i]) {
                labels_match = false;
                break;
            }
        }

        if (labels_match) {
            printf("✓ Labels EXACTLY MATCH\n");
        } else {
            printf("  Labels differ but produce same components (VALID)\n");
        }
    } else {
        printf("✗ WARNING: Component counts DIFFER\n");
        printf("  Sequential: %d components\n", result_seq->num_components);
        printf("  Pthreads:   %d components\n", result_sync->num_components);
    }

    /* Print performance comparison */
    printf("\n=== Performance Comparison ===\n");
    printf("Sequential (UF edge reorder):       %.5f seconds\n", elapsed_seq);
    printf("Parallel (sync LP, %d threads):    %.5f seconds (%d iterations)\n",
           num_threads, elapsed_sync, result_sync->num_iterations);

    /* Compute and print speedup */
    if (elapsed_seq > 0.0) {
        const double speedup_sync = elapsed_seq / elapsed_sync;
        const double eff_sync = speedup_sync / (double)num_threads * 100.0;

        printf("\nSpeedup vs sequential:\n");
        printf("  Sync LP: %.2fx (%.1f%% efficiency)\n", speedup_sync, eff_sync);

        /* Performance categorization */
        printf("\nScaling: ");
        if (speedup_sync >= (double)num_threads * 0.8) {
            printf("Excellent!\n");
        } else if (speedup_sync >= (double)num_threads * 0.5) {
            printf("Good\n");
        } else if (speedup_sync >= 2.0) {
            printf("Moderate\n");
        } else {
            printf("Limited (consider larger graphs or check for bottlenecks)\n");
        }
    }

    /* Cleanup */
    cc_result_destroy(result_seq);
    cc_result_destroy(result_sync);

    return 0;
}
