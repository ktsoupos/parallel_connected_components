#include "benchmark.h"
#include "cc_cuda.h"
#include "cc_sequential.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

/* Helper function to get wall time */
static double get_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int run_parallel_benchmarks(const Graph* g, int num_threads) {
    (void)num_threads;  /* CUDA doesn't use num_threads parameter */

    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return -1;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);

    /* Check CUDA device availability */
    printf("\n=== CUDA Device Check ===\n");
    if (cuda_check_device() != 0) {
        fprintf(stderr, "Error: No CUDA devices available\n");
        return -1;
    }

    printf("\n=== Parallel Benchmarks (CUDA/GPU) ===\n");

    /* Run sequential baseline for comparison */
    printf("\n=== Sequential Baseline (Union-Find Edge Reorder) ===\n");
    const double start_seq = get_wall_time();
    CCResult* result_seq = union_find_cc_edge_reorder(g);
    const double end_seq = get_wall_time();

    if (result_seq == NULL) {
        fprintf(stderr, "Error: Sequential algorithm failed\n");
        return -1;
    }

    const double elapsed_seq = end_seq - start_seq;
    printf("Sequential UF completed in %.5f seconds\n", elapsed_seq);
    cc_result_print_stats(result_seq, g);

    /* Run CUDA label propagation */
    printf("\n=== CUDA Label Propagation ===\n");
    const double start_cuda = get_wall_time();
    CCResult* result_cuda = cc_cuda(g);
    const double end_cuda = get_wall_time();

    if (result_cuda == NULL) {
        fprintf(stderr, "Error: CUDA label propagation failed\n");
        cc_result_destroy(result_seq);
        return -1;
    }

    const double elapsed_cuda = end_cuda - start_cuda;
    printf("CUDA LP completed in %.5f seconds\n", elapsed_cuda);
    cc_result_print_stats(result_cuda, g);

    /* Run ECL-CC algorithm */
    printf("\n=== CUDA ECL-CC ===\n");
    const double start_ecl = get_wall_time();
    CCResult* result_ecl = cc_cuda_ecl(g);
    const double end_ecl = get_wall_time();

    if (result_ecl == NULL) {
        fprintf(stderr, "Error: ECL-CC algorithm failed\n");
        cc_result_destroy(result_seq);
        cc_result_destroy(result_cuda);
        return -1;
    }

    const double elapsed_ecl = end_ecl - start_ecl;
    printf("ECL-CC completed in %.5f seconds\n", elapsed_ecl);
    cc_result_print_stats(result_ecl, g);

    /* Run Afforest algorithm */
    printf("\n=== CUDA Afforest ===\n");
    const double start_afforest = get_wall_time();
    CCResult* result_afforest = cc_cuda_afforest(g);
    const double end_afforest = get_wall_time();

    if (result_afforest == NULL) {
        fprintf(stderr, "Error: Afforest algorithm failed\n");
        cc_result_destroy(result_seq);
        cc_cuda_result_destroy(result_cuda);
        cc_cuda_result_destroy(result_ecl);
        return -1;
    }

    const double elapsed_afforest = end_afforest - start_afforest;
    printf("Afforest completed in %.5f seconds\n", elapsed_afforest);
    cc_result_print_stats(result_afforest, g);

    /* Verify correctness: compare component counts */
    printf("\n=== Correctness Verification ===\n");

    /* Verify CUDA LP */
    if (result_seq->num_components == result_cuda->num_components) {
        printf("CUDA LP:   Component counts MATCH (%d components)\n", result_seq->num_components);
    } else {
        printf("WARNING: CUDA LP component counts DIFFER\n");
        printf("  Sequential: %d, CUDA LP: %d\n", result_seq->num_components, result_cuda->num_components);
    }

    /* Verify ECL-CC */
    if (result_seq->num_components == result_ecl->num_components) {
        printf("ECL-CC:    Component counts MATCH (%d components)\n", result_seq->num_components);
    } else {
        printf("WARNING: ECL-CC component counts DIFFER\n");
        printf("  Sequential: %d, ECL-CC: %d\n", result_seq->num_components, result_ecl->num_components);
    }

    /* Verify Afforest */
    if (result_seq->num_components == result_afforest->num_components) {
        printf("Afforest:  Component counts MATCH (%d components)\n", result_seq->num_components);
    } else {
        printf("WARNING: Afforest component counts DIFFER\n");
        printf("  Sequential: %d, Afforest: %d\n", result_seq->num_components, result_afforest->num_components);
    }

    /* Print performance comparison */
    printf("\n=== Performance Comparison ===\n");
    printf("Sequential (UF edge reorder): %.5f seconds\n", elapsed_seq);
    printf("CUDA (Label Propagation):     %.5f seconds (%d iterations)\n",
           elapsed_cuda, result_cuda->num_iterations);
    printf("CUDA (ECL-CC):                %.5f seconds\n", elapsed_ecl);
    printf("CUDA (Afforest):              %.5f seconds\n", elapsed_afforest);

    /* Compute and print speedups */
    printf("\n=== Speedup vs Sequential ===\n");
    if (elapsed_seq > 0.0) {
        const double speedup_cuda = elapsed_seq / elapsed_cuda;
        const double speedup_ecl = elapsed_seq / elapsed_ecl;
        const double speedup_afforest = elapsed_seq / elapsed_afforest;
        printf("CUDA LP:   %.2fx\n", speedup_cuda);
        printf("ECL-CC:    %.2fx\n", speedup_ecl);
        printf("Afforest:  %.2fx\n", speedup_afforest);

        printf("\n=== GPU Algorithm Comparison ===\n");
        if (elapsed_ecl > 0.0 && elapsed_afforest > 0.0) {
            printf("ECL-CC vs CUDA LP:    %.2fx\n", elapsed_cuda / elapsed_ecl);
            printf("Afforest vs CUDA LP:  %.2fx\n", elapsed_cuda / elapsed_afforest);
            printf("Afforest vs ECL-CC:   %.2fx\n", elapsed_ecl / elapsed_afforest);
        }
    }

    /* Cleanup */
    cc_result_destroy(result_seq);
    cc_cuda_result_destroy(result_cuda);
    cc_cuda_result_destroy(result_ecl);
    cc_result_destroy(result_afforest);

    return 0;
}
