#include "cc_mpi.h"
#include "cc_common.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/**
 * Buffer for organizing remote edge requests to a specific process
 */
typedef struct {
    int32_t *local_vertices; // Local indices that need remote linking
    int32_t *global_vertices; // Global IDs to query from this process
    int32_t *parent_values; // Parent values received back
    int32_t count; // Number of requests
    int32_t capacity; // Allocated space
} RemoteBuffer;

/**
 * MPI communication metadata for Alltoallv operations
 */
typedef struct {
    int32_t *send_counts; // [num_ranks] - items to send to each rank
    int32_t *recv_counts; // [num_ranks] - items to receive from each rank
    int32_t *send_displs; // [num_ranks] - displacements for sending
    int32_t *recv_displs; // [num_ranks] - displacements for receiving
    int32_t total_send; // Total items to send
    int32_t total_recv; // Total items to receive
} CommData;


/* ============ GRAPH PARTITIONING ============ */

int partition_graph(const Graph *global_graph,
                    DistributedGraph **dist_graph,
                    MPI_Comm comm) {
    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    int32_t meta[2] = {0, 0};
    if (rank == 0) {
        meta[0] = global_graph->num_vertices;
        meta[1] = global_graph->num_edges;
    }

    MPI_Bcast(meta, 2, MPI_INT32_T, 0, comm);

    const int32_t g_num_vertices = meta[0];
    const int32_t g_num_edges = meta[1];
    const int32_t verts_per_proc = g_num_vertices / num_ranks;
    const int32_t vertex_offset = rank * verts_per_proc;

    int32_t l_num_vertices =
        (rank == num_ranks - 1)
            ? (g_num_vertices - vertex_offset)
            : verts_per_proc;

    *dist_graph = calloc(1, sizeof(DistributedGraph));
    if (!*dist_graph)
        return -1;

    DistributedGraph *dg = *dist_graph;

    dg->g_num_vertices = g_num_vertices;
    dg->g_num_edges = g_num_edges;
    dg->l_num_vertices = l_num_vertices;
    dg->vertex_offset = vertex_offset;
    dg->rank = rank;
    dg->num_ranks = num_ranks;
    dg->comm = comm;


    dg->local_row_ptr = malloc(sizeof(int32_t) * (l_num_vertices + 1));
    if (!dg->local_row_ptr)
        return -1;

    int32_t *row_sendcounts = NULL;
    int32_t *row_displs = NULL;

    if (rank == 0) {
        row_sendcounts = malloc(num_ranks * sizeof(int32_t));
        row_displs = malloc(num_ranks * sizeof(int32_t));

        for (int i = 0; i < num_ranks; i++) {
            row_sendcounts[i] =
                (i == num_ranks - 1)
                    ? (g_num_vertices - i * verts_per_proc + 1)
                    : (verts_per_proc + 1);

            row_displs[i] = i * verts_per_proc;
        }
    }

    MPI_Scatterv(
        rank == 0 ? global_graph->row_ptr : NULL,
        row_sendcounts,
        row_displs,
        MPI_INT32_T,
        dg->local_row_ptr,
        l_num_vertices + 1,
        MPI_INT32_T,
        0,
        comm
        );

    if (rank == 0) {
        free(row_sendcounts);
        free(row_displs);
    }

    int32_t l_num_edges = 0;
    int32_t *edge_sendcounts = NULL;
    int32_t *edge_displs = NULL;

    if (rank == 0) {
        edge_sendcounts = malloc(num_ranks * sizeof(int32_t));
        edge_displs = malloc(num_ranks * sizeof(int32_t));

        for (int i = 0; i < num_ranks; i++) {
            int32_t v_offset = i * verts_per_proc;
            int32_t v_count =
                (i == num_ranks - 1)
                    ? (g_num_vertices - v_offset)
                    : verts_per_proc;

            int32_t e_start = global_graph->row_ptr[v_offset];
            int32_t e_end = global_graph->row_ptr[v_offset + v_count];

            edge_sendcounts[i] = e_end - e_start;
            edge_displs[i] = e_start;
        }
    }

    MPI_Scatter(
        edge_sendcounts,
        1,
        MPI_INT32_T,
        &l_num_edges,
        1,
        MPI_INT32_T,
        0,
        comm
        );

    dg->l_num_edges = l_num_edges;

    dg->local_col_idx = malloc(sizeof(int32_t) * l_num_edges);
    if (!dg->local_col_idx)
        return -1;

    MPI_Scatterv(
        rank == 0 ? global_graph->col_idx : NULL,
        edge_sendcounts,
        edge_displs,
        MPI_INT32_T,
        dg->local_col_idx,
        l_num_edges,
        MPI_INT32_T,
        0,
        comm
        );

    if (rank == 0) {
        free(edge_sendcounts);
        free(edge_displs);
    }

    int32_t base = dg->local_row_ptr[0];
    for (int i = 0; i <= l_num_vertices; i++) {
        dg->local_row_ptr[i] -= base;
    }

    return 0;
}

/* ============ MPI LABEL PROPAGATION ============ */

/**
 * MPI Label Propagation for Connected Components
 *
 * Algorithm:
 * 1. Each process maintains labels for its local vertices
 * 2. Each iteration:
 *    - Exchange all labels globally using MPI_Allgatherv
 *    - For each local vertex, propagate minimum label from neighbors
 *    - Check global convergence (no changes anywhere)
 * 3. Continue until convergence
 *
 * @param dg - Distributed graph structure
 * @return CCResult with component labels, or NULL on error
 */
CCResult *mpi_label_propagation(const DistributedGraph *dg) {
    if (dg == NULL) {
        fprintf(stderr, "Error: NULL DistributedGraph pointer\n");
        return NULL;
    }

    const int rank = dg->rank;
    const int num_ranks = dg->num_ranks;
    const int32_t g_num_vertices = dg->g_num_vertices;
    const int32_t l_num_vertices = dg->l_num_vertices;
    const int32_t vertex_offset = dg->vertex_offset;

    /* Allocate result structure */
    CCResult *result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate CCResult\n", rank);
        return NULL;
    }

    /* Allocate global labels array (all processes need all labels) */
    int32_t *global_labels = malloc(sizeof(int32_t) * (size_t)g_num_vertices);
    if (global_labels == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate global labels\n", rank);
        free(result);
        return NULL;
    }

    /* Allocate local labels array */
    int32_t *local_labels = malloc(sizeof(int32_t) * (size_t)l_num_vertices);
    if (local_labels == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate local labels\n", rank);
        free(global_labels);
        free(result);
        return NULL;
    }

    /* Initialize local labels: each vertex gets its global ID as initial label */
    for (int32_t i = 0; i < l_num_vertices; i++) {
        local_labels[i] = vertex_offset + i;
    }

    /* Prepare MPI_Allgatherv parameters */
    int32_t *recvcounts = malloc(sizeof(int32_t) * (size_t)num_ranks);
    int32_t *displs = malloc(sizeof(int32_t) * (size_t)num_ranks);
    if (recvcounts == NULL || displs == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate MPI communication arrays\n", rank);
        free(local_labels);
        free(global_labels);
        free(recvcounts);
        free(displs);
        free(result);
        return NULL;
    }

    /* Calculate vertex counts and displacements for each rank */
    const int32_t verts_per_proc = g_num_vertices / num_ranks;
    for (int i = 0; i < num_ranks; i++) {
        displs[i] = i * verts_per_proc;
        recvcounts[i] = (i == num_ranks - 1)
                        ? (g_num_vertices - displs[i])
                        : verts_per_proc;
    }

    /* Label propagation iterations */
    result->num_iterations = 0;
    bool global_changed = true;

    while (global_changed) {
        result->num_iterations++;

        /* Exchange labels: gather all local labels to global_labels on all processes */
        int mpi_result = MPI_Allgatherv(
            local_labels,           /* Send buffer: local labels */
            l_num_vertices,         /* Send count */
            MPI_INT32_T,           /* Send type */
            global_labels,          /* Receive buffer: all labels */
            recvcounts,             /* Receive counts per rank */
            displs,                 /* Displacements per rank */
            MPI_INT32_T,           /* Receive type */
            dg->comm                /* Communicator */
        );

        if (mpi_result != MPI_SUCCESS) {
            fprintf(stderr, "Rank %d: MPI_Allgatherv failed at iteration %d\n",
                    rank, result->num_iterations);
            free(displs);
            free(recvcounts);
            free(local_labels);
            free(global_labels);
            free(result);
            return NULL;
        }

        /* Propagate labels locally */
        bool local_changed = false;

        for (int32_t i = 0; i < l_num_vertices; i++) {
            const int32_t global_v = vertex_offset + i;
            int32_t min_label = global_labels[global_v];

            /* Find minimum label among neighbors */
            const int32_t edge_start = dg->local_row_ptr[i];
            const int32_t edge_end = dg->local_row_ptr[i + 1];

            for (int32_t e = edge_start; e < edge_end; e++) {
                const int32_t neighbor = dg->local_col_idx[e];

                /* Bounds check for safety */
                if (neighbor < 0 || neighbor >= g_num_vertices) {
                    fprintf(stderr, "Rank %d: Invalid neighbor %d for vertex %d\n",
                            rank, neighbor, global_v);
                    continue;
                }

                if (global_labels[neighbor] < min_label) {
                    min_label = global_labels[neighbor];
                }
            }

            /* Update if label changed */
            if (min_label < local_labels[i]) {
                local_labels[i] = min_label;
                local_changed = true;
            }
        }

        /* Check global convergence: any process has changes? */
        int local_changed_int = local_changed ? 1 : 0;
        int global_changed_int = 0;

        mpi_result = MPI_Allreduce(
            &local_changed_int,     /* Send buffer */
            &global_changed_int,    /* Receive buffer */
            1,                      /* Count */
            MPI_INT,               /* Type */
            MPI_LOR,               /* Logical OR operation */
            dg->comm               /* Communicator */
        );

        if (mpi_result != MPI_SUCCESS) {
            fprintf(stderr, "Rank %d: MPI_Allreduce failed at iteration %d\n",
                    rank, result->num_iterations);
            free(displs);
            free(recvcounts);
            free(local_labels);
            free(global_labels);
            free(result);
            return NULL;
        }

        global_changed = (global_changed_int != 0);
    }

    /* Gather final labels on rank 0 */
    MPI_Allgatherv(
        local_labels,
        l_num_vertices,
        MPI_INT32_T,
        global_labels,
        recvcounts,
        displs,
        MPI_INT32_T,
        dg->comm
    );

    /* Only rank 0 needs to return the full result */
    if (rank == 0) {
        result->labels = global_labels;

        /* Count unique components */
        result->num_components = count_unique_labels(global_labels, g_num_vertices);
    } else {
        result->labels = NULL;
        result->num_components = 0;
        free(global_labels);
    }

    /* Cleanup */
    free(local_labels);
    free(recvcounts);
    free(displs);

    return result;
}


