#include "benchmark.h"
#include "afforest_simple.h"
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

    printf("\n=== Parallel Benchmarks (Afforest with %d threads) ===\n", num_threads);

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

    /* Run Afforest Simple with pthreads */
    printf("\n=== Parallel Afforest (Pthreads) ===\n");
    clock_t start_afforest_simple = clock();
    CCResult* result_afforest_simple = afforest_simple_pthreads(g, num_threads, 2);
    clock_t end_afforest_simple = clock();

    if (result_afforest_simple == NULL) {
        fprintf(stderr, "Error: Afforest failed\n");
        cc_result_destroy(result_seq);
        return -1;
    }

    const double elapsed_afforest_simple = (double)(end_afforest_simple - start_afforest_simple) / CLOCKS_PER_SEC;
    printf("Parallel Afforest completed in %.5f seconds\n", elapsed_afforest_simple);
    cc_result_print_stats(result_afforest_simple, g);

    /* Verify correctness: compare component counts */
    printf("\n=== Correctness Verification ===\n");
    bool afforest_correct = (result_seq->num_components == result_afforest_simple->num_components);

    if (afforest_correct) {
        printf("✓ Both algorithms found %d components\n", result_seq->num_components);
    } else {
        printf("✗ WARNING: Component counts DIFFER\n");
        printf("  Sequential: %d components\n", result_seq->num_components);
        printf("  Parallel:   %d components ✗\n", result_afforest_simple->num_components);
    }

    /* Print performance comparison */
    printf("\n=== Performance Comparison ===\n");
    printf("Sequential (UF edge reorder):       %.5f seconds\n", elapsed_seq);
    printf("Parallel (Afforest, %d threads):    %.5f seconds (%d iterations)\n",
           num_threads, elapsed_afforest_simple, result_afforest_simple->num_iterations);

    /* Compute and print speedup */
    if (elapsed_seq > 0.0) {
        const double speedup = elapsed_seq / elapsed_afforest_simple;
        const double efficiency = speedup / (double)num_threads * 100.0;

        printf("\nSpeedup vs sequential: %.2fx (%.1f%% efficiency)\n", speedup, efficiency);

        /* Performance categorization */
        printf("Parallel scaling: ");
        if (speedup >= (double)num_threads * 0.8) {
            printf("Excellent!\n");
        } else if (speedup >= (double)num_threads * 0.5) {
            printf("Good\n");
        } else if (speedup >= 2.0) {
            printf("Moderate\n");
        } else {
            printf("Limited (consider larger graphs or check for bottlenecks)\n");
        }
    }

    /* Cleanup */
    cc_result_destroy(result_seq);
    cc_result_destroy(result_afforest_simple);

    return 0;
}
