#include "benchmark.h"
#include "cc_mpi.h"
#include "cc_sequential.h"
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int run_mpi_benchmarks(const Graph *g) {
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (g == NULL && rank == 0) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return -1;
    }

    if (rank == 0) {
        printf("\n=== MPI Distributed Benchmarks (%d processes) ===\n", num_ranks);
        printf("Graph: %d vertices, %d edges\n",
               graph_get_num_vertices(g), graph_get_num_edges(g));
    }

    /* Sequential baseline (rank 0 only, for comparison) */
    CCResult *result_seq = NULL;
    double elapsed_seq = 0.0;

    if (rank == 0) {
        printf("\n=== Sequential Baseline (Union-Find Edge Reorder) ===\n");
        const double start_seq = MPI_Wtime();
        result_seq = union_find_cc_edge_reorder(g);
        const double end_seq = MPI_Wtime();

        if (result_seq == NULL) {
            fprintf(stderr, "Error: Sequential algorithm failed\n");
            return -1;
        }

        elapsed_seq = end_seq - start_seq;
        printf("Sequential UF completed in %.5f seconds\n", elapsed_seq);
        cc_result_print_stats(result_seq, g);
    }

    /* Partition graph across all processes */
    DistributedGraph *dist_graph = NULL;
    const int partition_result = partition_graph(g, &dist_graph, MPI_COMM_WORLD);

    if (partition_result != 0) {
        fprintf(stderr, "Rank %d: Error partitioning graph\n", rank);
        if (rank == 0 && result_seq != NULL) {
            cc_result_destroy(result_seq);
        }
        return -1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* MPI Shiloach-Vishkin Algorithm */
    if (rank == 0) {
        printf("\n=== MPI Shiloach-Vishkin Connected Components ===\n");
    }

    const double start_sv = MPI_Wtime();
    CCResult *result_sv = NULL;
    const double end_sv = MPI_Wtime();

    if (result_sv == NULL) {
        fprintf(stderr, "Rank %d: Error in Shiloach-Vishkin algorithm\n", rank);
        free(dist_graph->local_row_ptr);
        free(dist_graph->local_col_idx);
        free(dist_graph);
        if (rank == 0 && result_seq != NULL) {
            cc_result_destroy(result_seq);
        }
        return -1;
    }

    const double elapsed_sv = end_sv - start_sv;

    if (rank == 0) {
        printf("MPI Shiloach-Vishkin completed in %.5f seconds\n", elapsed_sv);
        printf("  Components: %d\n", result_sv->num_components);
        printf("  Iterations: %d\n", result_sv->num_iterations);
        printf("  Speedup vs Sequential: %.2fx\n", elapsed_seq / elapsed_sv);
    }

    /* Performance Summary */
    if (rank == 0) {
        printf("\n=== MPI Performance Summary ===\n");
        printf("%-25s %12s %12s %12s\n", "Algorithm", "Time (s)", "Components", "Speedup");
        printf("%-25s %12.5f %12d %12s\n",
               "Sequential UF", elapsed_seq, result_seq->num_components, "1.00x");
        printf("%-25s %12.5f %12d %12.2fx\n",
               "MPI Shiloach-Vishkin", elapsed_sv, result_sv->num_components,
               elapsed_seq / elapsed_sv);
    }

    /* Cleanup */
    if (rank == 0 && result_seq != NULL) {
        cc_result_destroy(result_seq);
    }
    cc_result_destroy(result_sv);
    free(dist_graph->local_row_ptr);
    free(dist_graph->local_col_idx);
    free(dist_graph);

    return 0;
}
