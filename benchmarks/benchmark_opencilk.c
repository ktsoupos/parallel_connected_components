#include "benchmark_opencilk.h"
#include "cc_opencilk.h"
#include "cc_sequential.h"
#include <cilk/cilk.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

/**
 * Get high-resolution wall clock time in seconds
 */
static double get_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int run_opencilk_benchmarks(const Graph* g, int num_workers) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return -1;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);

    printf("\n=== Parallel Benchmarks (OpenCilk with %d workers) ===\n", num_workers);

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

    /* Run Afforest algorithm with Cilk */
    printf("\n=== OpenCilk Afforest Connected Components ===\n");
    const double start_afforest = get_wall_time();
    CCResult* result_afforest = afforest_cilk(g, num_workers, 2);
    const double end_afforest = get_wall_time();

    if (result_afforest == NULL) {
        fprintf(stderr, "Error: Afforest algorithm failed\n");
        cc_result_destroy(result_seq);
        return -1;
    }

    const double elapsed_afforest = end_afforest - start_afforest;
    printf("OpenCilk Afforest completed in %.5f seconds\n", elapsed_afforest);
    cc_result_print_stats(result_afforest, g);



    /* Run Recursive Edge-Based Union-Find */
    printf("\n=== OpenCilk Recursive Edge-Based Union-Find ===\n");
    const double start_recursive = get_wall_time();
    CCResult* result_recursive = recursive_edge_cc(g, num_workers);
    const double end_recursive = get_wall_time();

    if (result_recursive == NULL) {
        fprintf(stderr, "Error: Recursive Edge UF algorithm failed\n");
        cc_result_destroy(result_seq);
        cc_result_destroy(result_afforest);
        return -1;
    }

    const double elapsed_recursive = end_recursive - start_recursive;
    printf("OpenCilk Recursive Edge UF completed in %.5f seconds\n", elapsed_recursive);
    cc_result_print_stats(result_recursive, g);

    /* Verify correctness: compare component counts */
    printf("\n=== Correctness Verification ===\n");
    if (result_seq->num_components == result_afforest->num_components &&
        result_seq->num_components == result_recursive->num_components) {
        printf("✓ Component counts MATCH: %d components\n", result_seq->num_components);
        printf("  Sequential:     %d\n", result_seq->num_components);
        printf("  Afforest:       %d\n", result_afforest->num_components);
        printf("  Recursive Edge: %d\n", result_recursive->num_components);
    } else {
        printf("✗ WARNING: Component counts DIFFER\n");
        printf("  Sequential:     %d components\n", result_seq->num_components);
        printf("  Afforest:       %d components\n", result_afforest->num_components);
        printf("  Recursive Edge: %d components\n", result_recursive->num_components);
    }

    /* Print performance comparison */
    printf("\n=== Performance Comparison ===\n");
    printf("Sequential (UF edge reorder):         %.5f seconds\n", elapsed_seq);
    printf("Parallel (Afforest, %d workers):     %.5f seconds\n",
           num_workers, elapsed_afforest);
    printf("Parallel (Recursive Edge, %d workers): %.5f seconds\n",
           num_workers, elapsed_recursive);

    /* Compute and print speedup */
    if (elapsed_seq > 0.0) {
        const double speedup_afforest = elapsed_seq / elapsed_afforest;
        const double eff_afforest = speedup_afforest / (double)num_workers * 100.0;


        const double speedup_recursive = elapsed_seq / elapsed_recursive;
        const double eff_recursive = speedup_recursive / (double)num_workers * 100.0;

        printf("\nSpeedup vs sequential:\n");
        printf("  Afforest:       %.2fx (%.1f%% efficiency)\n", speedup_afforest, eff_afforest);
        printf("  Recursive Edge: %.2fx (%.1f%% efficiency)\n", speedup_recursive, eff_recursive);

        /* Performance categorization for Afforest */
        printf("\nAfforest scaling:       ");
        if (speedup_afforest >= (double)num_workers * 0.8) {
            printf("Excellent!\n");
        } else if (speedup_afforest >= (double)num_workers * 0.5) {
            printf("Good\n");
        } else if (speedup_afforest >= 2.0) {
            printf("Moderate\n");
        } else {
            printf("Limited (consider larger graphs)\n");
        }

        /* Performance categorization for Recursive Edge */
        printf("Recursive Edge scaling: ");
        if (speedup_recursive >= (double)num_workers * 0.8) {
            printf("Excellent!\n");
        } else if (speedup_recursive >= (double)num_workers * 0.5) {
            printf("Good\n");
        } else if (speedup_recursive >= 2.0) {
            printf("Moderate\n");
        } else {
            printf("Limited (consider larger graphs)\n");
        }
    }

    /* Cleanup */
    cc_result_destroy(result_seq);
    cc_result_destroy(result_afforest);
    cc_result_destroy(result_recursive);

    return 0;
}
