#include "benchmark.h"
#include "afforest_simple.h"
#include "cc_pthreads.h"
#include "cc_sequential.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L  /* For clock_gettime */
#endif

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
    CCResult* result_seq = union_find_cc_edge_reorder(g);
    clock_t end_seq = clock();

    if (result_seq == NULL) {
        fprintf(stderr, "Error: Sequential algorithm failed\n");
        return -1;
    }

    const double elapsed_seq = (double)(end_seq - start_seq) / CLOCKS_PER_SEC;
    printf("Sequential UF completed in %.5f seconds\n", elapsed_seq);
    cc_result_print_stats(result_seq, g);

    /* Run Afforest Simple with pthreads - STATIC SCHEDULING */
    printf("\n=== Parallel Afforest (Pthreads - Static Scheduling) ===\n");
    struct timespec start_static, end_static;
    clock_gettime(CLOCK_MONOTONIC, &start_static);
    CCResult* result_static = afforest_simple_pthreads(g, num_threads, 2, false);
    clock_gettime(CLOCK_MONOTONIC, &end_static);

    if (result_static == NULL) {
        fprintf(stderr, "Error: Afforest static failed\n");
        cc_result_destroy(result_seq);
        return -1;
    }

    const double elapsed_static = (end_static.tv_sec - start_static.tv_sec) +
                                  (end_static.tv_nsec - start_static.tv_nsec) / 1e9;
    printf("Static scheduling completed in %.5f seconds\n", elapsed_static);
    cc_result_print_stats(result_static, g);

    /* Run Afforest Simple with pthreads - DYNAMIC SCHEDULING */
    printf("\n=== Parallel Afforest (Pthreads - Dynamic Scheduling) ===\n");
    struct timespec start_dynamic, end_dynamic;
    clock_gettime(CLOCK_MONOTONIC, &start_dynamic);
    CCResult* result_dynamic = afforest_simple_pthreads(g, num_threads, 2, true);
    clock_gettime(CLOCK_MONOTONIC, &end_dynamic);

    if (result_dynamic == NULL) {
        fprintf(stderr, "Error: Afforest dynamic failed\n");
        cc_result_destroy(result_seq);
        cc_result_destroy(result_static);
        return -1;
    }

    const double elapsed_dynamic = (end_dynamic.tv_sec - start_dynamic.tv_sec) +
                                   (end_dynamic.tv_nsec - start_dynamic.tv_nsec) / 1e9;
    printf("Dynamic scheduling completed in %.5f seconds\n", elapsed_dynamic);
    cc_result_print_stats(result_dynamic, g);

    /* Verify correctness: compare component counts */
    printf("\n=== Correctness Verification ===\n");
    bool static_correct = (result_seq->num_components == result_static->num_components);
    bool dynamic_correct = (result_seq->num_components == result_dynamic->num_components);

    if (static_correct && dynamic_correct) {
        printf("✓ All algorithms found %d components\n", result_seq->num_components);
    } else {
        printf("✗ WARNING: Component counts DIFFER\n");
        printf("  Sequential: %d components\n", result_seq->num_components);
        printf("  Static:     %d components %s\n", result_static->num_components, static_correct ? "✓" : "✗");
        printf("  Dynamic:    %d components %s\n", result_dynamic->num_components, dynamic_correct ? "✓" : "✗");
    }

    /* Print performance comparison */
    printf("\n=== Performance Comparison ===\n");
    printf("Sequential (UF edge reorder):           %.5f seconds\n", elapsed_seq);
    printf("Parallel Static  (Afforest, %d threads): %.5f seconds (%d iterations)\n",
           num_threads, elapsed_static, result_static->num_iterations);
    printf("Parallel Dynamic (Afforest, %d threads): %.5f seconds (%d iterations)\n",
           num_threads, elapsed_dynamic, result_dynamic->num_iterations);

    /* Compute speedups for static and dynamic (will use later) */
    double speedup_static = 0.0, speedup_dynamic = 0.0;
    double efficiency_static = 0.0, efficiency_dynamic = 0.0;

    if (elapsed_seq > 0.0) {
        speedup_static = elapsed_seq / elapsed_static;
        speedup_dynamic = elapsed_seq / elapsed_dynamic;
        efficiency_static = speedup_static / (double)num_threads * 100.0;
        efficiency_dynamic = speedup_dynamic / (double)num_threads * 100.0;

        printf("\nStatic  speedup: %.2fx (%.1f%% efficiency)\n", speedup_static, efficiency_static);
        printf("Dynamic speedup: %.2fx (%.1f%% efficiency)\n", speedup_dynamic, efficiency_dynamic);

        /* Dynamic vs Static comparison */
        if (elapsed_dynamic < elapsed_static) {
            const double improvement = (elapsed_static / elapsed_dynamic - 1.0) * 100.0;
            printf("Dynamic is %.1f%% FASTER than static\n", improvement);
        } else {
            const double slowdown = (elapsed_dynamic / elapsed_static - 1.0) * 100.0;
            printf("Dynamic is %.1f%% slower than static\n", slowdown);
        }
    }

    /* Run Label Propagation Sync (simplified pthreads) */
    printf("\n=== Parallel Label Propagation (Simple Pthreads) ===\n");
    struct timespec start_lp, end_lp;
    clock_gettime(CLOCK_MONOTONIC, &start_lp);
    CCResult* result_lp = label_propagation_sync_pthreads(g, num_threads);
    clock_gettime(CLOCK_MONOTONIC, &end_lp);

    if (result_lp == NULL) {
        fprintf(stderr, "Error: Label propagation failed\n");
        cc_result_destroy(result_seq);
        cc_result_destroy(result_static);
        cc_result_destroy(result_dynamic);
        return -1;
    }

    const double elapsed_lp = (end_lp.tv_sec - start_lp.tv_sec) +
                              (end_lp.tv_nsec - start_lp.tv_nsec) / 1e9;
    printf("Label Propagation completed in %.5f seconds\n", elapsed_lp);
    cc_result_print_stats(result_lp, g);

    /* Verify correctness for all algorithms */
    printf("\n=== Correctness Verification (All Algorithms) ===\n");
    bool lp_correct = (result_seq->num_components == result_lp->num_components);

    if (static_correct && dynamic_correct && lp_correct) {
        printf("✓ All algorithms found %d components\n", result_seq->num_components);
    } else {
        printf("✗ WARNING: Component counts DIFFER\n");
        printf("  Sequential:         %d components\n", result_seq->num_components);
        printf("  Afforest Static:    %d components %s\n", result_static->num_components, static_correct ? "✓" : "✗");
        printf("  Afforest Dynamic:   %d components %s\n", result_dynamic->num_components, dynamic_correct ? "✓" : "✗");
        printf("  Label Propagation:  %d components %s\n", result_lp->num_components, lp_correct ? "✓" : "✗");
    }

    /* Print comprehensive performance comparison */
    printf("\n=== Complete Performance Comparison ===\n");
    printf("Sequential (UF edge reorder):              %.5f seconds\n", elapsed_seq);
    printf("Afforest Static   (%d threads):            %.5f seconds (%d iterations)\n",
           num_threads, elapsed_static, result_static->num_iterations);
    printf("Afforest Dynamic  (%d threads):            %.5f seconds (%d iterations)\n",
           num_threads, elapsed_dynamic, result_dynamic->num_iterations);
    printf("Label Propagation (%d threads, Simple):    %.5f seconds (%d iterations)\n",
           num_threads, elapsed_lp, result_lp->num_iterations);

    /* Compute speedups for all parallel versions */
    if (elapsed_seq > 0.0) {
        const double speedup_lp = elapsed_seq / elapsed_lp;
        const double efficiency_lp = speedup_lp / (double)num_threads * 100.0;

        printf("\n=== Speedups vs Sequential ===\n");
        printf("Afforest Static:     %.2fx (%.1f%% efficiency)\n", speedup_static, efficiency_static);
        printf("Afforest Dynamic:    %.2fx (%.1f%% efficiency)\n", speedup_dynamic, efficiency_dynamic);
        printf("Label Propagation:   %.2fx (%.1f%% efficiency)\n", speedup_lp, efficiency_lp);

        /* Find fastest algorithm */
        printf("\n=== Algorithm Ranking ===\n");
        double times[] = {elapsed_static, elapsed_dynamic, elapsed_lp};
        const char* names[] = {"Afforest Static", "Afforest Dynamic", "Label Propagation"};

        double fastest_time = times[0];
        int fastest_idx = 0;
        for (int i = 1; i < 3; i++) {
            if (times[i] < fastest_time) {
                fastest_time = times[i];
                fastest_idx = i;
            }
        }

        printf("FASTEST: %s (%.5f seconds)\n", names[fastest_idx], fastest_time);
        for (int i = 0; i < 3; i++) {
            if (i != fastest_idx) {
                double slowdown = (times[i] / fastest_time - 1.0) * 100.0;
                printf("  %s: %.1f%% slower\n", names[i], slowdown);
            }
        }
    }

    /* Cleanup */
    cc_result_destroy(result_seq);
    cc_result_destroy(result_static);
    cc_result_destroy(result_dynamic);
    cc_result_destroy(result_lp);

    return 0;
}
