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
        fflush(stdout);
    }

    /* Sequential baselines (rank 0 only, for comparison) */
    CCResult *result_seq_uf = NULL;
    CCResult *result_seq_lp = NULL;
    double elapsed_seq_uf = 0.0;
    double elapsed_seq_lp = 0.0;

    if (rank == 0) {
        printf("\n=== Sequential Baseline (Union-Find Edge Reorder) ===\n");
        fflush(stdout);
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
        fflush(stdout);
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
    if (rank == 0) {
        printf("\nPartitioning graph across %d processes...\n", num_ranks);
        fflush(stdout);
    }

    DistributedGraph *dist_graph = NULL;
    const int partition_result = partition_graph(g, &dist_graph, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Partitioning complete!\n");
        printf("  Local vertices per process: ~%d\n", dist_graph->l_num_vertices);
        printf("  Local edges per process: ~%d\n", dist_graph->l_num_edges);
        fflush(stdout);
    }

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
        fflush(stdout);
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
        if (dist_graph->num_ghost_vertices >= 0) {
            printf("  Ghost vertices: %d (%.1f%% of local)\n",
                   dist_graph->num_ghost_vertices,
                   100.0 * dist_graph->num_ghost_vertices / dist_graph->l_num_vertices);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* MPI Label Propagation Simple Async (MPI_Iallgatherv) */
    if (rank == 0) {
        printf("\n=== MPI Label Propagation Simple Async (MPI_Iallgatherv) ===\n");
        fflush(stdout);
    }

    const double start_lp_simple = MPI_Wtime();
    CCResult *result_mpi_lp_simple = mpi_label_propagation_simple_async(dist_graph);
    const double end_lp_simple = MPI_Wtime();

    if (result_mpi_lp_simple == NULL) {
        fprintf(stderr, "Rank %d: Error in MPI Label Propagation Simple Async algorithm\n", rank);
        cc_result_destroy(result_mpi_lp);
        free(dist_graph->local_row_ptr);
        free(dist_graph->local_col_idx);
        free(dist_graph);
        if (rank == 0) {
            if (result_seq_uf != NULL) cc_result_destroy(result_seq_uf);
            if (result_seq_lp != NULL) cc_result_destroy(result_seq_lp);
        }
        return -1;
    }

    const double elapsed_mpi_lp_simple = end_lp_simple - start_lp_simple;

    if (rank == 0) {
        printf("MPI LP Simple Async completed in %.5f seconds\n", elapsed_mpi_lp_simple);
        printf("  Components: %d\n", result_mpi_lp_simple->num_components);
        printf("  Iterations: %d\n", result_mpi_lp_simple->num_iterations);
        printf("  Speedup vs Sequential LP: %.2fx\n", elapsed_seq_lp / elapsed_mpi_lp_simple);
        printf("  Speedup vs Sequential UF: %.2fx\n", elapsed_seq_uf / elapsed_mpi_lp_simple);
        printf("  Speedup vs MPI LP (basic): %.2fx\n", elapsed_mpi_lp / elapsed_mpi_lp_simple);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* MPI Label Propagation Optimized (Ghost Exchange) */
    if (rank == 0) {
        printf("\n=== MPI Label Propagation Optimized (Ghost/Halo Exchange + Async) ===\n");
        fflush(stdout);
    }

    const double start_lp_opt = MPI_Wtime();
    CCResult *result_mpi_lp_opt = mpi_label_propagation_optimized(dist_graph);
    const double end_lp_opt = MPI_Wtime();

    if (result_mpi_lp_opt == NULL) {
        fprintf(stderr, "Rank %d: Error in MPI Label Propagation Optimized algorithm\n", rank);
        cc_result_destroy(result_mpi_lp);
        free(dist_graph->local_row_ptr);
        free(dist_graph->local_col_idx);
        /* Free ghost structures */
        free(dist_graph->ghost_global_ids);
        free(dist_graph->ghost_to_owner);
        free(dist_graph->ghost_labels);
        free(dist_graph->send_counts);
        free(dist_graph->recv_counts);
        free(dist_graph->send_displs);
        free(dist_graph->recv_displs);
        for (int r = 0; r < num_ranks; r++) {
            free(dist_graph->send_vertices[r]);
        }
        free(dist_graph->send_vertices);
        free(dist_graph);
        if (rank == 0) {
            if (result_seq_uf != NULL) cc_result_destroy(result_seq_uf);
            if (result_seq_lp != NULL) cc_result_destroy(result_seq_lp);
        }
        return -1;
    }

    const double elapsed_mpi_lp_opt = end_lp_opt - start_lp_opt;

    if (rank == 0) {
        printf("MPI LP Optimized completed in %.5f seconds\n", elapsed_mpi_lp_opt);
        printf("  Components: %d\n", result_mpi_lp_opt->num_components);
        printf("  Iterations: %d\n", result_mpi_lp_opt->num_iterations);
        printf("  Speedup vs Sequential LP: %.2fx\n", elapsed_seq_lp / elapsed_mpi_lp_opt);
        printf("  Speedup vs Sequential UF: %.2fx\n", elapsed_seq_uf / elapsed_mpi_lp_opt);
        printf("  Speedup vs MPI LP (basic): %.2fx\n", elapsed_mpi_lp / elapsed_mpi_lp_opt);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* MPI Label Propagation Fully Async (Progressive Boundary Processing) */
    if (rank == 0) {
        printf("\n=== MPI Label Propagation Fully Async (Progressive Boundary + MPI_Testsome) ===\n");
        fflush(stdout);
    }

    const double start_lp_async = MPI_Wtime();
    CCResult *result_mpi_lp_async = mpi_label_propagation_async(dist_graph);
    const double end_lp_async = MPI_Wtime();

    if (result_mpi_lp_async == NULL) {
        fprintf(stderr, "Rank %d: Error in MPI Label Propagation Async algorithm\n", rank);
        cc_result_destroy(result_mpi_lp);
        cc_result_destroy(result_mpi_lp_opt);
        free(dist_graph->local_row_ptr);
        free(dist_graph->local_col_idx);
        free(dist_graph->ghost_global_ids);
        free(dist_graph->ghost_to_owner);
        free(dist_graph->ghost_labels);
        free(dist_graph->send_counts);
        free(dist_graph->recv_counts);
        free(dist_graph->send_displs);
        free(dist_graph->recv_displs);
        for (int r = 0; r < num_ranks; r++) {
            free(dist_graph->send_vertices[r]);
        }
        free(dist_graph->send_vertices);
        free(dist_graph);
        if (rank == 0) {
            if (result_seq_uf != NULL) cc_result_destroy(result_seq_uf);
            if (result_seq_lp != NULL) cc_result_destroy(result_seq_lp);
        }
        return -1;
    }

    const double elapsed_mpi_lp_async = end_lp_async - start_lp_async;

    if (rank == 0) {
        printf("MPI LP Fully Async completed in %.5f seconds\n", elapsed_mpi_lp_async);
        printf("  Components: %d\n", result_mpi_lp_async->num_components);
        printf("  Iterations: %d\n", result_mpi_lp_async->num_iterations);
        printf("  Speedup vs Sequential LP: %.2fx\n", elapsed_seq_lp / elapsed_mpi_lp_async);
        printf("  Speedup vs Sequential UF: %.2fx\n", elapsed_seq_uf / elapsed_mpi_lp_async);
        printf("  Speedup vs MPI LP (basic): %.2fx\n", elapsed_mpi_lp / elapsed_mpi_lp_async);
        printf("  Speedup vs MPI LP Optimized: %.2fx\n", elapsed_mpi_lp_opt / elapsed_mpi_lp_async);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* MPI Union-Find */
    if (rank == 0) {
        printf("\n=== MPI Union-Find Connected Components ===\n");
        fflush(stdout);
    }

    const double start_uf = MPI_Wtime();
    CCResult *result_mpi_uf = mpi_union_find_cc(dist_graph);
    const double end_uf = MPI_Wtime();

    if (result_mpi_uf == NULL) {
        fprintf(stderr, "Rank %d: Error in MPI Union-Find algorithm\n", rank);
        cc_result_destroy(result_mpi_lp);
        cc_result_destroy(result_mpi_lp_simple);
        cc_result_destroy(result_mpi_lp_opt);
        cc_result_destroy(result_mpi_lp_async);
        free(dist_graph->local_row_ptr);
        free(dist_graph->local_col_idx);
        free(dist_graph->ghost_global_ids);
        free(dist_graph->ghost_to_owner);
        free(dist_graph->ghost_labels);
        free(dist_graph->send_counts);
        free(dist_graph->recv_counts);
        free(dist_graph->send_displs);
        free(dist_graph->recv_displs);
        for (int r = 0; r < num_ranks; r++) {
            free(dist_graph->send_vertices[r]);
        }
        free(dist_graph->send_vertices);
        free(dist_graph);
        if (rank == 0) {
            if (result_seq_uf != NULL) cc_result_destroy(result_seq_uf);
            if (result_seq_lp != NULL) cc_result_destroy(result_seq_lp);
        }
        return -1;
    }

    const double elapsed_mpi_uf = end_uf - start_uf;

    if (rank == 0) {
        printf("MPI Union-Find completed in %.5f seconds\n", elapsed_mpi_uf);
        printf("  Components: %d\n", result_mpi_uf->num_components);
        printf("  Iterations: %d\n", result_mpi_uf->num_iterations);
        printf("  Speedup vs Sequential LP: %.2fx\n", elapsed_seq_lp / elapsed_mpi_uf);
        printf("  Speedup vs Sequential UF: %.2fx\n", elapsed_seq_uf / elapsed_mpi_uf);
        printf("  Speedup vs MPI LP (basic): %.2fx\n", elapsed_mpi_lp / elapsed_mpi_uf);
    }

    /* Performance Summary */
    if (rank == 0) {
        printf("\n=== MPI Performance Summary ===\n");
        printf("%-45s %12s %12s %12s\n", "Algorithm", "Time (s)", "Components", "Speedup");
        printf("%-45s %12.5f %12d %12s\n",
               "Sequential UF", elapsed_seq_uf, result_seq_uf->num_components, "1.00x");
        printf("%-45s %12.5f %12d %12.2fx\n",
               "Sequential LP", elapsed_seq_lp, result_seq_lp->num_components,
               elapsed_seq_uf / elapsed_seq_lp);
        printf("%-45s %12.5f %12d %12.2fx\n",
               "MPI LP (sync Allgatherv)", elapsed_mpi_lp, result_mpi_lp->num_components,
               elapsed_seq_uf / elapsed_mpi_lp);
        printf("%-45s %12.5f %12d %12.2fx\n",
               "MPI LP Simple Async (Iallgatherv)", elapsed_mpi_lp_simple, result_mpi_lp_simple->num_components,
               elapsed_seq_uf / elapsed_mpi_lp_simple);
        printf("%-45s %12.5f %12d %12.2fx\n",
               "MPI LP Optimized (Ghost+Async)", elapsed_mpi_lp_opt, result_mpi_lp_opt->num_components,
               elapsed_seq_uf / elapsed_mpi_lp_opt);
        printf("%-45s %12.5f %12d %12.2fx\n",
               "MPI LP Fully Async (Ghost+MPI_Testsome)", elapsed_mpi_lp_async, result_mpi_lp_async->num_components,
               elapsed_seq_uf / elapsed_mpi_lp_async);
        printf("%-45s %12.5f %12d %12.2fx\n",
               "MPI Union-Find (Alltoallv)", elapsed_mpi_uf, result_mpi_uf->num_components,
               elapsed_seq_uf / elapsed_mpi_uf);
    }

    /* Cleanup */
    if (rank == 0) {
        if (result_seq_uf != NULL) cc_result_destroy(result_seq_uf);
        if (result_seq_lp != NULL) cc_result_destroy(result_seq_lp);
    }
    cc_result_destroy(result_mpi_lp);
    cc_result_destroy(result_mpi_lp_simple);
    cc_result_destroy(result_mpi_lp_opt);
    cc_result_destroy(result_mpi_lp_async);
    cc_result_destroy(result_mpi_uf);

    /* Free distributed graph and ghost structures */
    free(dist_graph->local_row_ptr);
    free(dist_graph->local_col_idx);
    free(dist_graph->ghost_global_ids);
    free(dist_graph->ghost_to_owner);
    free(dist_graph->ghost_labels);
    free(dist_graph->send_counts);
    free(dist_graph->recv_counts);
    free(dist_graph->send_displs);
    free(dist_graph->recv_displs);
    if (dist_graph->send_vertices != NULL) {
        for (int r = 0; r < num_ranks; r++) {
            free(dist_graph->send_vertices[r]);
        }
        free(dist_graph->send_vertices);
    }
    free(dist_graph);

    return 0;
}
