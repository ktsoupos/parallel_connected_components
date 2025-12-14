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

    /* Sequential baselines (rank 0 only, for comparison) */
    CCResult *result_seq_uf = NULL;
    CCResult *result_seq_lp = NULL;
    double elapsed_seq_uf = 0.0;
    double elapsed_seq_lp = 0.0;

    if (rank == 0) {
        printf("\n=== Sequential Baseline (Union-Find Edge Reorder) ===\n");
        const double start_seq = MPI_Wtime();
        result_seq_uf = union_find_cc_edge_reorder(g);
        const double end_seq = MPI_Wtime();

        if (result_seq_uf == NULL) {
            fprintf(stderr, "Error: Sequential Union-Find algorithm failed\n");
            return -1;
        }

        elapsed_seq_uf = end_seq - start_seq;
        printf("Sequential UF completed in %.5f seconds\n", elapsed_seq_uf);
        cc_result_print_stats(result_seq_uf, g);

        printf("\n=== Sequential Baseline (Label Propagation) ===\n");
        const double start_lp = MPI_Wtime();
        result_seq_lp = label_propagation_min(g);
        const double end_lp = MPI_Wtime();

        if (result_seq_lp == NULL) {
            fprintf(stderr, "Error: Sequential Label Propagation algorithm failed\n");
            cc_result_destroy(result_seq_uf);
            return -1;
        }

        elapsed_seq_lp = end_lp - start_lp;
        printf("Sequential LP completed in %.5f seconds\n", elapsed_seq_lp);
        cc_result_print_stats(result_seq_lp, g);
    }

    /* Partition graph across all processes */
    DistributedGraph *dist_graph = NULL;
    const int partition_result = partition_graph(g, &dist_graph, MPI_COMM_WORLD);

    if (partition_result != 0) {
        fprintf(stderr, "Rank %d: Error partitioning graph\n", rank);
        if (rank == 0) {
            if (result_seq_uf != NULL) cc_result_destroy(result_seq_uf);
            if (result_seq_lp != NULL) cc_result_destroy(result_seq_lp);
        }
        return -1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* MPI Label Propagation Algorithm */
    if (rank == 0) {
        printf("\n=== MPI Label Propagation Connected Components ===\n");
    }

    const double start_lp = MPI_Wtime();
    CCResult *result_mpi_lp = mpi_label_propagation(dist_graph);
    const double end_lp = MPI_Wtime();

    if (result_mpi_lp == NULL) {
        fprintf(stderr, "Rank %d: Error in MPI Label Propagation algorithm\n", rank);
        free(dist_graph->local_row_ptr);
        free(dist_graph->local_col_idx);
        free(dist_graph);
        if (rank == 0) {
            if (result_seq_uf != NULL) cc_result_destroy(result_seq_uf);
            if (result_seq_lp != NULL) cc_result_destroy(result_seq_lp);
        }
        return -1;
    }

    const double elapsed_mpi_lp = end_lp - start_lp;

    if (rank == 0) {
        printf("MPI Label Propagation completed in %.5f seconds\n", elapsed_mpi_lp);
        printf("  Components: %d\n", result_mpi_lp->num_components);
        printf("  Iterations: %d\n", result_mpi_lp->num_iterations);
        printf("  Speedup vs Sequential LP: %.2fx\n", elapsed_seq_lp / elapsed_mpi_lp);
        printf("  Speedup vs Sequential UF: %.2fx\n", elapsed_seq_uf / elapsed_mpi_lp);
    }

    /* Performance Summary */
    if (rank == 0) {
        printf("\n=== MPI Performance Summary ===\n");
        printf("%-30s %12s %12s %12s\n", "Algorithm", "Time (s)", "Components", "Speedup");
        printf("%-30s %12.5f %12d %12s\n",
               "Sequential UF", elapsed_seq_uf, result_seq_uf->num_components, "1.00x");
        printf("%-30s %12.5f %12d %12.2fx\n",
               "Sequential LP", elapsed_seq_lp, result_seq_lp->num_components,
               elapsed_seq_uf / elapsed_seq_lp);
        printf("%-30s %12.5f %12d %12.2fx\n",
               "MPI Label Propagation", elapsed_mpi_lp, result_mpi_lp->num_components,
               elapsed_seq_uf / elapsed_mpi_lp);
    }

    /* Cleanup */
    if (rank == 0) {
        if (result_seq_uf != NULL) cc_result_destroy(result_seq_uf);
        if (result_seq_lp != NULL) cc_result_destroy(result_seq_lp);
    }
    cc_result_destroy(result_mpi_lp);
    free(dist_graph->local_row_ptr);
    free(dist_graph->local_col_idx);
    free(dist_graph);

    return 0;
}
