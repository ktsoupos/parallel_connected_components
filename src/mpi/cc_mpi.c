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

    /* Build ghost vertex map for communication optimization */
    /* First pass: identify unique ghost vertices */
    bool *is_ghost = calloc((size_t)g_num_vertices, sizeof(bool));
    if (!is_ghost) {
        fprintf(stderr, "Rank %d: Failed to allocate is_ghost array\n", rank);
        return -1;
    }

    /* Mark all remote neighbors as ghosts */
    for (int32_t i = 0; i < l_num_vertices; i++) {
        const int32_t edge_start = dg->local_row_ptr[i];
        const int32_t edge_end = dg->local_row_ptr[i + 1];

        for (int32_t e = edge_start; e < edge_end; e++) {
            const int32_t neighbor = dg->local_col_idx[e];

            /* Check if neighbor is outside local range */
            if (neighbor < vertex_offset || neighbor >= vertex_offset + l_num_vertices) {
                is_ghost[neighbor] = true;
            }
        }
    }

    /* Count ghost vertices */
    dg->num_ghost_vertices = 0;
    for (int32_t v = 0; v < g_num_vertices; v++) {
        if (is_ghost[v]) {
            dg->num_ghost_vertices++;
        }
    }

    /* Allocate ghost arrays */
    if (dg->num_ghost_vertices > 0) {
        dg->ghost_global_ids = malloc(sizeof(int32_t) * (size_t)dg->num_ghost_vertices);
        dg->ghost_to_owner = malloc(sizeof(int32_t) * (size_t)dg->num_ghost_vertices);
        dg->ghost_labels = malloc(sizeof(int32_t) * (size_t)dg->num_ghost_vertices);

        if (!dg->ghost_global_ids || !dg->ghost_to_owner || !dg->ghost_labels) {
            fprintf(stderr, "Rank %d: Failed to allocate ghost arrays\n", rank);
            free(is_ghost);
            return -1;
        }

        /* Fill ghost arrays temporarily in global ID order */
        int32_t ghost_idx = 0;
        for (int32_t v = 0; v < g_num_vertices; v++) {
            if (is_ghost[v]) {
                dg->ghost_global_ids[ghost_idx] = v;
                /* Determine owner rank */
                int owner_rank = v / verts_per_proc;
                if (owner_rank >= num_ranks) owner_rank = num_ranks - 1;
                dg->ghost_to_owner[ghost_idx] = owner_rank;
                ghost_idx++;
            }
        }

        /* CRITICAL: Sort ghost vertices by owner rank for correct Alltoallv communication */
        int32_t *temp_global_ids = malloc(sizeof(int32_t) * (size_t)dg->num_ghost_vertices);
        int32_t *temp_owners = malloc(sizeof(int32_t) * (size_t)dg->num_ghost_vertices);
        int32_t *rank_counts = calloc((size_t)num_ranks, sizeof(int32_t));
        int32_t *rank_offsets = malloc(sizeof(int32_t) * (size_t)num_ranks);

        if (!temp_global_ids || !temp_owners || !rank_counts || !rank_offsets) {
            fprintf(stderr, "Rank %d: Failed to allocate temp sort arrays\n", rank);
            free(temp_global_ids);
            free(temp_owners);
            free(rank_counts);
            free(rank_offsets);
            free(is_ghost);
            return -1;
        }

        /* Count ghosts per rank */
        for (int32_t i = 0; i < dg->num_ghost_vertices; i++) {
            rank_counts[dg->ghost_to_owner[i]]++;
        }

        /* Calculate offsets */
        rank_offsets[0] = 0;
        for (int r = 1; r < num_ranks; r++) {
            rank_offsets[r] = rank_offsets[r - 1] + rank_counts[r - 1];
        }

        /* Reset counts for reuse as current positions */
        for (int r = 0; r < num_ranks; r++) {
            rank_counts[r] = rank_offsets[r];
        }

        /* Reorder by owner rank */
        for (int32_t i = 0; i < dg->num_ghost_vertices; i++) {
            int owner = dg->ghost_to_owner[i];
            int32_t pos = rank_counts[owner]++;
            temp_global_ids[pos] = dg->ghost_global_ids[i];
            temp_owners[pos] = owner;
        }

        /* Copy back sorted arrays */
        for (int32_t i = 0; i < dg->num_ghost_vertices; i++) {
            dg->ghost_global_ids[i] = temp_global_ids[i];
            dg->ghost_to_owner[i] = temp_owners[i];
        }

        free(temp_global_ids);
        free(temp_owners);
        free(rank_counts);
        free(rank_offsets);
    } else {
        dg->ghost_global_ids = NULL;
        dg->ghost_to_owner = NULL;
        dg->ghost_labels = NULL;
    }

    free(is_ghost);

    /* Build send/receive communication metadata */
    dg->send_counts = calloc((size_t)num_ranks, sizeof(int32_t));
    dg->recv_counts = calloc((size_t)num_ranks, sizeof(int32_t));
    dg->send_displs = calloc((size_t)num_ranks, sizeof(int32_t));
    dg->recv_displs = calloc((size_t)num_ranks, sizeof(int32_t));
    dg->send_vertices = calloc((size_t)num_ranks, sizeof(int32_t *));

    if (!dg->send_counts || !dg->recv_counts || !dg->send_displs ||
        !dg->recv_displs || !dg->send_vertices) {
        fprintf(stderr, "Rank %d: Failed to allocate communication metadata\n", rank);
        return -1;
    }

    /* Count how many ghost vertices we receive from each rank (now sorted by owner) */
    for (int32_t i = 0; i < dg->num_ghost_vertices; i++) {
        int owner = dg->ghost_to_owner[i];
        dg->recv_counts[owner]++;
    }

    /* Exchange recv_counts to get send_counts (transpose) */
    MPI_Alltoall(dg->recv_counts, 1, MPI_INT32_T,
                 dg->send_counts, 1, MPI_INT32_T, comm);

    /* Calculate displacements */
    dg->send_displs[0] = 0;
    dg->recv_displs[0] = 0;
    for (int r = 1; r < num_ranks; r++) {
        dg->send_displs[r] = dg->send_displs[r - 1] + dg->send_counts[r - 1];
        dg->recv_displs[r] = dg->recv_displs[r - 1] + dg->recv_counts[r - 1];
    }

    /* Allocate send_vertices arrays for each rank */
    for (int r = 0; r < num_ranks; r++) {
        if (dg->send_counts[r] > 0) {
            dg->send_vertices[r] = malloc(sizeof(int32_t) * (size_t)dg->send_counts[r]);
            if (!dg->send_vertices[r]) {
                fprintf(stderr, "Rank %d: Failed to allocate send_vertices[%d]\n", rank, r);
                return -1;
            }
        } else {
            dg->send_vertices[r] = NULL;
        }
    }

    /* Exchange global IDs to determine which local vertices to send */
    int32_t *recv_requests = NULL;
    if (dg->num_ghost_vertices > 0) {
        recv_requests = malloc(sizeof(int32_t) * (size_t)dg->num_ghost_vertices);
        if (!recv_requests) {
            fprintf(stderr, "Rank %d: Failed to allocate recv_requests\n", rank);
            return -1;
        }
        for (int32_t i = 0; i < dg->num_ghost_vertices; i++) {
            recv_requests[i] = dg->ghost_global_ids[i];
        }
    }

    int32_t total_send = dg->send_displs[num_ranks - 1] + dg->send_counts[num_ranks - 1];
    int32_t *send_requests = malloc(sizeof(int32_t) * (size_t)(total_send > 0 ? total_send : 1));
    if (!send_requests) {
        fprintf(stderr, "Rank %d: Failed to allocate send_requests\n", rank);
        free(recv_requests);
        return -1;
    }

    MPI_Alltoallv(recv_requests, dg->recv_counts, dg->recv_displs, MPI_INT32_T,
                  send_requests, dg->send_counts, dg->send_displs, MPI_INT32_T, comm);

    /* Convert global IDs to local indices */
    for (int r = 0; r < num_ranks; r++) {
        for (int32_t i = 0; i < dg->send_counts[r]; i++) {
            int32_t global_id = send_requests[dg->send_displs[r] + i];
            dg->send_vertices[r][i] = global_id - vertex_offset;
        }
    }

    free(send_requests);
    free(recv_requests);

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

/* ============ SIMPLE ASYNC MPI LABEL PROPAGATION ============ */

/**
 * Simple Asynchronous MPI Label Propagation
 *
 * Uses MPI_Iallgatherv (non-blocking collective) instead of MPI_Allgatherv
 * to overlap communication with computation where possible.
 *
 * No ghost vertices - simpler than optimized version but still async.
 *
 * @param dg - Distributed graph structure
 * @return CCResult with component labels, or NULL on error
 */
CCResult *mpi_label_propagation_simple_async(const DistributedGraph *dg) {
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

    /* Prepare MPI_Iallgatherv parameters */
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

    /* Label propagation iterations with async allgatherv */
    result->num_iterations = 0;
    bool global_changed = true;
    MPI_Request gather_request = MPI_REQUEST_NULL;
    bool gather_in_flight = false;

    while (global_changed) {
        result->num_iterations++;

        /* Start non-blocking allgatherv for previous iteration's results */
        if (result->num_iterations > 1 && gather_in_flight) {
            /* Wait for previous gather to complete before starting new one */
            MPI_Wait(&gather_request, MPI_STATUS_IGNORE);
            gather_in_flight = false;
        }

        /* Start async gather of current labels */
        int mpi_result = MPI_Iallgatherv(
            local_labels,           /* Send buffer: local labels */
            l_num_vertices,         /* Send count */
            MPI_INT32_T,           /* Send type */
            global_labels,          /* Receive buffer: all labels */
            recvcounts,             /* Receive counts per rank */
            displs,                 /* Displacements per rank */
            MPI_INT32_T,           /* Receive type */
            dg->comm,              /* Communicator */
            &gather_request         /* Request handle */
        );

        if (mpi_result != MPI_SUCCESS) {
            fprintf(stderr, "Rank %d: MPI_Iallgatherv failed at iteration %d\n",
                    rank, result->num_iterations);
            free(displs);
            free(recvcounts);
            free(local_labels);
            free(global_labels);
            free(result);
            return NULL;
        }
        gather_in_flight = true;

        /* Wait for gather to complete before we can use global_labels */
        MPI_Wait(&gather_request, MPI_STATUS_IGNORE);
        gather_in_flight = false;

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

/* ============ OPTIMIZED MPI LABEL PROPAGATION WITH GHOST EXCHANGE ============ */

/**
 * Asynchronous ghost label exchange with fine-grained completion tracking
 * Starts non-blocking send/receive operations and tracks individual recv completion
 *
 * @param dg - Distributed graph with ghost metadata
 * @param local_labels - Local vertex labels
 * @param recv_requests - Output array for receive MPI_Request handles
 * @param send_requests - Output array for send MPI_Request handles
 * @param num_recv_requests - Output number of receive requests
 * @param num_send_requests - Output number of send requests
 * @param send_buffers_out - Output send buffers (must be freed after completion)
 * @param recv_from_rank - Output array mapping recv request index to source rank
 * @return 0 on success, -1 on error
 */
static int start_ghost_exchange_async(const DistributedGraph *dg,
                                       const int32_t *local_labels,
                                       MPI_Request **recv_requests,
                                       MPI_Request **send_requests,
                                       int *num_recv_requests,
                                       int *num_send_requests,
                                       int32_t ***send_buffers_out,
                                       int **recv_from_rank) {
    const int num_ranks = dg->num_ranks;
    const int rank = dg->rank;

    /* Count actual send and recv operations */
    int num_sends = 0;
    int num_recvs = 0;
    for (int r = 0; r < num_ranks; r++) {
        if (r != rank) {
            if (dg->send_counts[r] > 0) num_sends++;
            if (dg->recv_counts[r] > 0) num_recvs++;
        }
    }

    *num_recv_requests = num_recvs;
    *num_send_requests = num_sends;

    if (num_recvs == 0 && num_sends == 0) {
        *recv_requests = NULL;
        *send_requests = NULL;
        *send_buffers_out = NULL;
        *recv_from_rank = NULL;
        return 0;
    }

    /* Allocate separate arrays for recv and send requests */
    if (num_recvs > 0) {
        *recv_requests = malloc(sizeof(MPI_Request) * (size_t)num_recvs);
        *recv_from_rank = malloc(sizeof(int) * (size_t)num_recvs);
        if (*recv_requests == NULL || *recv_from_rank == NULL) {
            fprintf(stderr, "Rank %d: Failed to allocate recv request arrays\n", rank);
            free(*recv_requests);
            free(*recv_from_rank);
            return -1;
        }
    } else {
        *recv_requests = NULL;
        *recv_from_rank = NULL;
    }

    if (num_sends > 0) {
        *send_requests = malloc(sizeof(MPI_Request) * (size_t)num_sends);
        if (*send_requests == NULL) {
            fprintf(stderr, "Rank %d: Failed to allocate send request array\n", rank);
            free(*recv_requests);
            free(*recv_from_rank);
            return -1;
        }
    } else {
        *send_requests = NULL;
    }

    /* Allocate send buffers (one per rank) */
    int32_t **send_buffers = malloc(sizeof(int32_t *) * (size_t)num_ranks);
    if (send_buffers == NULL) {
        free(*recv_requests);
        free(*send_requests);
        free(*recv_from_rank);
        return -1;
    }

    for (int r = 0; r < num_ranks; r++) {
        if (dg->send_counts[r] > 0 && r != rank) {
            send_buffers[r] = malloc(sizeof(int32_t) * (size_t)dg->send_counts[r]);
            if (send_buffers[r] == NULL) {
                /* Cleanup on error */
                for (int j = 0; j < r; j++) {
                    free(send_buffers[j]);
                }
                free(send_buffers);
                free(*recv_requests);
                free(*send_requests);
                free(*recv_from_rank);
                return -1;
            }

            /* Pack labels to send */
            for (int32_t i = 0; i < dg->send_counts[r]; i++) {
                int32_t local_idx = dg->send_vertices[r][i];
                send_buffers[r][i] = local_labels[local_idx];
            }
        } else {
            send_buffers[r] = NULL;
        }
    }

    /* Post receives first */
    int recv_idx = 0;
    for (int r = 0; r < num_ranks; r++) {
        if (r == rank) continue;

        if (dg->recv_counts[r] > 0) {
            MPI_Irecv(
                dg->ghost_labels + dg->recv_displs[r],
                dg->recv_counts[r],
                MPI_INT32_T,
                r,
                0, /* tag */
                dg->comm,
                &(*recv_requests)[recv_idx]
            );
            (*recv_from_rank)[recv_idx] = r;
            recv_idx++;
        }
    }

    /* Post sends */
    int send_idx = 0;
    for (int r = 0; r < num_ranks; r++) {
        if (r == rank) continue;

        if (dg->send_counts[r] > 0) {
            MPI_Isend(
                send_buffers[r],
                dg->send_counts[r],
                MPI_INT32_T,
                r,
                0, /* tag */
                dg->comm,
                &(*send_requests)[send_idx++]
            );
        }
    }

    /* Return send buffers so caller can free after MPI_Wait */
    *send_buffers_out = send_buffers;

    return 0;
}

/**
 * Asynchronous ghost label exchange
 * Starts non-blocking send/receive operations for boundary vertices only
 *
 * @param dg - Distributed graph with ghost metadata
 * @param local_labels - Local vertex labels
 * @param requests - Output array for MPI_Request handles
 * @param num_requests - Output number of active requests
 * @param send_buffers_out - Output send buffers (must be freed after MPI_Waitall)
 * @return 0 on success, -1 on error
 */
static int start_ghost_exchange(const DistributedGraph *dg,
                                 const int32_t *local_labels,
                                 MPI_Request **requests,
                                 int *num_requests,
                                 int32_t ***send_buffers_out) {
    const int num_ranks = dg->num_ranks;
    const int rank = dg->rank;

    /* Count actual send and recv operations */
    int num_sends = 0;
    int num_recvs = 0;
    for (int r = 0; r < num_ranks; r++) {
        if (r != rank) {
            if (dg->send_counts[r] > 0) num_sends++;
            if (dg->recv_counts[r] > 0) num_recvs++;
        }
    }

    *num_requests = num_sends + num_recvs;
    if (*num_requests == 0) {
        *requests = NULL;
        *send_buffers_out = NULL;
        return 0;
    }

    *requests = malloc(sizeof(MPI_Request) * (size_t)(*num_requests));
    if (*requests == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate request array\n", rank);
        return -1;
    }

    /* Allocate send buffers (one per rank) */
    int32_t **send_buffers = malloc(sizeof(int32_t *) * (size_t)num_ranks);
    if (send_buffers == NULL) {
        free(*requests);
        *requests = NULL;
        return -1;
    }

    for (int r = 0; r < num_ranks; r++) {
        if (dg->send_counts[r] > 0 && r != rank) {
            send_buffers[r] = malloc(sizeof(int32_t) * (size_t)dg->send_counts[r]);
            if (send_buffers[r] == NULL) {
                /* Cleanup on error */
                for (int j = 0; j < r; j++) {
                    free(send_buffers[j]);
                }
                free(send_buffers);
                free(*requests);
                *requests = NULL;
                return -1;
            }

            /* Pack labels to send */
            for (int32_t i = 0; i < dg->send_counts[r]; i++) {
                int32_t local_idx = dg->send_vertices[r][i];
                send_buffers[r][i] = local_labels[local_idx];
            }
        } else {
            send_buffers[r] = NULL;
        }
    }

    /* Start non-blocking send and receive operations */
    int req_idx = 0;
    for (int r = 0; r < num_ranks; r++) {
        if (r == rank) continue;

        /* Post receives */
        if (dg->recv_counts[r] > 0) {
            MPI_Irecv(
                dg->ghost_labels + dg->recv_displs[r],
                dg->recv_counts[r],
                MPI_INT32_T,
                r,
                0, /* tag */
                dg->comm,
                &(*requests)[req_idx++]
            );
        }

        /* Post sends */
        if (dg->send_counts[r] > 0) {
            MPI_Isend(
                send_buffers[r],
                dg->send_counts[r],
                MPI_INT32_T,
                r,
                0, /* tag */
                dg->comm,
                &(*requests)[req_idx++]
            );
        }
    }

    /* Return send buffers so caller can free after MPI_Wait */
    *send_buffers_out = send_buffers;

    return 0;
}

/**
 * Simple hash table for O(1) ghost vertex lookup
 */
typedef struct {
    int32_t *keys;      /* Global vertex IDs */
    int32_t *values;    /* Indices into ghost_labels */
    int32_t *next;      /* Collision chain */
    int32_t capacity;
    int32_t num_entries;
} GhostHashMap;

static GhostHashMap *create_ghost_hashmap(const DistributedGraph *dg) {
    GhostHashMap *map = malloc(sizeof(GhostHashMap));
    if (!map) return NULL;

    /* Use a hash table with capacity = 2 * num_ghosts for low collisions */
    map->capacity = dg->num_ghost_vertices * 2 + 1;
    if (map->capacity < 16) map->capacity = 16;

    map->keys = malloc(sizeof(int32_t) * (size_t)map->capacity);
    map->values = malloc(sizeof(int32_t) * (size_t)map->capacity);
    map->next = malloc(sizeof(int32_t) * (size_t)map->capacity);

    if (!map->keys || !map->values || !map->next) {
        free(map->keys);
        free(map->values);
        free(map->next);
        free(map);
        return NULL;
    }

    /* Initialize to empty */
    for (int32_t i = 0; i < map->capacity; i++) {
        map->keys[i] = -1;
        map->next[i] = -1;
    }

    /* Insert ghost vertices */
    map->num_entries = 0;
    for (int32_t i = 0; i < dg->num_ghost_vertices; i++) {
        int32_t key = dg->ghost_global_ids[i];
        int32_t hash = (key % map->capacity + map->capacity) % map->capacity;

        /* Find insertion point */
        if (map->keys[hash] == -1) {
            /* Empty slot */
            map->keys[hash] = key;
            map->values[hash] = i;
        } else {
            /* Collision - find free slot and chain */
            int32_t free_slot = (hash + 1) % map->capacity;
            int32_t attempts = 0;
            while (map->keys[free_slot] != -1) {
                free_slot = (free_slot + 1) % map->capacity;
                attempts++;
                if (attempts >= map->capacity) {
                    /* Hash map is full - should never happen */
                    fprintf(stderr, "ERROR: Hash map full during insertion!\n");
                    return NULL;
                }
            }
            map->keys[free_slot] = key;
            map->values[free_slot] = i;

            /* Chain from hash position */
            int32_t pos = hash;
            attempts = 0;
            while (map->next[pos] != -1) {
                pos = map->next[pos];
                attempts++;
                if (attempts >= map->capacity) {
                    /* Chain is circular - should never happen */
                    fprintf(stderr, "ERROR: Circular chain detected!\n");
                    return NULL;
                }
            }
            map->next[pos] = free_slot;
        }
        map->num_entries++;
    }

    return map;
}

static inline int32_t ghost_hashmap_get(const GhostHashMap *map, int32_t key) {
    int32_t hash = (key % map->capacity + map->capacity) % map->capacity;
    int32_t pos = hash;

    while (pos != -1) {
        if (map->keys[pos] == key) {
            return map->values[pos];
        }
        pos = map->next[pos];
    }

    return -1; /* Not found */
}

static void destroy_ghost_hashmap(GhostHashMap *map) {
    if (map) {
        free(map->keys);
        free(map->values);
        free(map->next);
        free(map);
    }
}

/**
 * Get label for a vertex (local or ghost) - optimized with binary search
 * Since ghost_global_ids is sorted, we can use binary search
 */
static inline int32_t get_vertex_label_fast(const DistributedGraph *dg,
                                             const int32_t *local_labels,
                                             const GhostHashMap *ghost_map,
                                             int32_t global_vertex_id) {
    const int32_t vertex_offset = dg->vertex_offset;
    const int32_t l_num_vertices = dg->l_num_vertices;

    /* Check if vertex is local */
    if (global_vertex_id >= vertex_offset &&
        global_vertex_id < vertex_offset + l_num_vertices) {
        return local_labels[global_vertex_id - vertex_offset];
    }

    /* Binary search in sorted ghost_global_ids array */
    int32_t left = 0;
    int32_t right = dg->num_ghost_vertices - 1;

    while (left <= right) {
        int32_t mid = left + (right - left) / 2;
        if (dg->ghost_global_ids[mid] == global_vertex_id) {
            return dg->ghost_labels[mid];
        } else if (dg->ghost_global_ids[mid] < global_vertex_id) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    /* Should never reach here if graph is correctly partitioned */
    fprintf(stderr, "Error: Vertex %d not found in local or ghost\n", global_vertex_id);
    return global_vertex_id; /* Fallback */
}

/**
 * Optimized MPI Label Propagation with Ghost/Halo Exchange
 *
 * Optimizations:
 * 1. Ghost exchange: Only communicate boundary vertices (not full graph)
 * 2. Asynchronous communication: Non-blocking MPI_Isend/Irecv
 * 3. Computation/communication overlap: Process interior vertices during communication
 *
 * @param dg - Distributed graph structure with ghost metadata
 * @return CCResult with component labels (rank 0 only), or NULL on error
 */
CCResult *mpi_label_propagation_optimized(const DistributedGraph *dg) {
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

    /* Allocate local labels (only local vertices) */
    int32_t *local_labels = malloc(sizeof(int32_t) * (size_t)l_num_vertices);
    if (local_labels == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate local labels\n", rank);
        free(result);
        return NULL;
    }

    /* Initialize labels: each vertex gets its global ID as initial label */
    for (int32_t i = 0; i < l_num_vertices; i++) {
        local_labels[i] = vertex_offset + i;
    }

    /* Initialize ghost labels */
    for (int32_t i = 0; i < dg->num_ghost_vertices; i++) {
        dg->ghost_labels[i] = dg->ghost_global_ids[i];
    }

    /* Identify boundary and interior vertices */
    bool *is_boundary = calloc((size_t)l_num_vertices, sizeof(bool));
    if (is_boundary == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate is_boundary array\n", rank);
        free(local_labels);
        free(result);
        return NULL;
    }

    for (int32_t i = 0; i < l_num_vertices; i++) {
        const int32_t edge_start = dg->local_row_ptr[i];
        const int32_t edge_end = dg->local_row_ptr[i + 1];

        for (int32_t e = edge_start; e < edge_end; e++) {
            const int32_t neighbor = dg->local_col_idx[e];
            /* If any neighbor is remote, this is a boundary vertex */
            if (neighbor < vertex_offset || neighbor >= vertex_offset + l_num_vertices) {
                is_boundary[i] = true;
                break;
            }
        }
    }

    /* No hash map needed - using binary search on sorted ghost_global_ids instead */
    GhostHashMap *ghost_map = NULL; /* Kept for function signature compatibility */

    /* Label propagation with async communication */
    result->num_iterations = 0;
    bool global_changed = true;

    while (global_changed) {
        result->num_iterations++;

        /* Start asynchronous ghost exchange */
        MPI_Request *requests = NULL;
        int num_requests = 0;
        int32_t **send_buffers = NULL;
        if (start_ghost_exchange(dg, local_labels, &requests, &num_requests, &send_buffers) != 0) {
            fprintf(stderr, "Rank %d: Ghost exchange failed\n", rank);
            destroy_ghost_hashmap(ghost_map);
            free(is_boundary);
            free(local_labels);
            free(result);
            return NULL;
        }

        /* Phase 1: Process interior vertices while communication is in flight */
        bool interior_changed = false;
        for (int32_t i = 0; i < l_num_vertices; i++) {
            if (is_boundary[i]) continue; /* Skip boundary, will process after communication */

            int32_t min_label = local_labels[i];

            const int32_t edge_start = dg->local_row_ptr[i];
            const int32_t edge_end = dg->local_row_ptr[i + 1];

            for (int32_t e = edge_start; e < edge_end; e++) {
                const int32_t neighbor = dg->local_col_idx[e];
                int32_t neighbor_label = local_labels[neighbor - vertex_offset]; /* All interior neighbors are local */

                if (neighbor_label < min_label) {
                    min_label = neighbor_label;
                }
            }

            if (min_label < local_labels[i]) {
                local_labels[i] = min_label;
                interior_changed = true;
            }
        }

        /* Wait for ghost exchange to complete */
        if (num_requests > 0) {
            MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
            free(requests);
        }

        /* Free send buffers now that MPI operations are complete */
        if (send_buffers != NULL) {
            for (int r = 0; r < num_ranks; r++) {
                free(send_buffers[r]);
            }
            free(send_buffers);
        }

        /* Phase 2: Process boundary vertices (now have updated ghost labels) */
        bool boundary_changed = false;
        for (int32_t i = 0; i < l_num_vertices; i++) {
            if (!is_boundary[i]) continue; /* Already processed interior */

            int32_t min_label = local_labels[i];

            const int32_t edge_start = dg->local_row_ptr[i];
            const int32_t edge_end = dg->local_row_ptr[i + 1];

            for (int32_t e = edge_start; e < edge_end; e++) {
                const int32_t neighbor = dg->local_col_idx[e];
                int32_t neighbor_label = get_vertex_label_fast(dg, local_labels, ghost_map, neighbor);

                if (neighbor_label < min_label) {
                    min_label = neighbor_label;
                }
            }

            if (min_label < local_labels[i]) {
                local_labels[i] = min_label;
                boundary_changed = true;
            }
        }

        /* Check global convergence */
        bool local_changed = interior_changed || boundary_changed;
        int local_changed_int = local_changed ? 1 : 0;
        int global_changed_int = 0;

        MPI_Allreduce(&local_changed_int, &global_changed_int, 1,
                      MPI_INT, MPI_LOR, dg->comm);

        global_changed = (global_changed_int != 0);
    }

    /* Gather final labels on all processes (for verification) */
    int32_t *all_labels = NULL;
    if (rank == 0) {
        all_labels = malloc(sizeof(int32_t) * (size_t)g_num_vertices);
        if (all_labels == NULL) {
            fprintf(stderr, "Rank %d: Failed to allocate all_labels\n", rank);
            free(is_boundary);
            free(local_labels);
            free(result);
            return NULL;
        }
    }

    /* Calculate gather parameters */
    int32_t *recvcounts = NULL;
    int32_t *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(sizeof(int32_t) * (size_t)num_ranks);
        displs = malloc(sizeof(int32_t) * (size_t)num_ranks);

        const int32_t verts_per_proc = g_num_vertices / num_ranks;
        for (int r = 0; r < num_ranks; r++) {
            displs[r] = r * verts_per_proc;
            recvcounts[r] = (r == num_ranks - 1)
                            ? (g_num_vertices - displs[r])
                            : verts_per_proc;
        }
    }

    MPI_Gatherv(local_labels, l_num_vertices, MPI_INT32_T,
                all_labels, recvcounts, displs, MPI_INT32_T,
                0, dg->comm);

    /* Only rank 0 returns the full result */
    if (rank == 0) {
        result->labels = all_labels;
        result->num_components = count_unique_labels(all_labels, g_num_vertices);
        free(recvcounts);
        free(displs);
    } else {
        result->labels = NULL;
        result->num_components = 0;
    }

    /* Cleanup */
    free(is_boundary);
    free(local_labels);

    return result;
}

/* ============ FULLY ASYNC MPI LABEL PROPAGATION ============ */

/**
 * Fully Asynchronous MPI Label Propagation with Progressive Boundary Processing
 *
 * Key improvements over optimized version:
 * 1. Uses MPI_Testsome to check individual receive completions
 * 2. Processes boundary vertices incrementally as their ghost data arrives
 * 3. Tracks which ghost vertices are ready per boundary vertex
 * 4. More aggressive computation/communication overlap
 *
 * @param dg - Distributed graph structure with ghost metadata
 * @return CCResult with component labels (rank 0 only), or NULL on error
 */
CCResult *mpi_label_propagation_async(const DistributedGraph *dg) {
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

    /* Allocate local labels (only local vertices) */
    int32_t *local_labels = malloc(sizeof(int32_t) * (size_t)l_num_vertices);
    if (local_labels == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate local labels\n", rank);
        free(result);
        return NULL;
    }

    /* Initialize labels: each vertex gets its global ID as initial label */
    for (int32_t i = 0; i < l_num_vertices; i++) {
        local_labels[i] = vertex_offset + i;
    }

    /* Initialize ghost labels */
    for (int32_t i = 0; i < dg->num_ghost_vertices; i++) {
        dg->ghost_labels[i] = dg->ghost_global_ids[i];
    }

    /* Identify boundary and interior vertices, track which ranks each boundary vertex depends on */
    bool *is_boundary = calloc((size_t)l_num_vertices, sizeof(bool));
    bool **boundary_needs_rank = malloc(sizeof(bool *) * (size_t)l_num_vertices);
    if (is_boundary == NULL || boundary_needs_rank == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate boundary tracking arrays\n", rank);
        free(local_labels);
        free(is_boundary);
        free(boundary_needs_rank);
        free(result);
        return NULL;
    }

    for (int32_t i = 0; i < l_num_vertices; i++) {
        boundary_needs_rank[i] = NULL;
    }

    for (int32_t i = 0; i < l_num_vertices; i++) {
        const int32_t edge_start = dg->local_row_ptr[i];
        const int32_t edge_end = dg->local_row_ptr[i + 1];

        for (int32_t e = edge_start; e < edge_end; e++) {
            const int32_t neighbor = dg->local_col_idx[e];
            /* If any neighbor is remote, this is a boundary vertex */
            if (neighbor < vertex_offset || neighbor >= vertex_offset + l_num_vertices) {
                if (!is_boundary[i]) {
                    is_boundary[i] = true;
                    boundary_needs_rank[i] = calloc((size_t)num_ranks, sizeof(bool));
                    if (boundary_needs_rank[i] == NULL) {
                        fprintf(stderr, "Rank %d: Failed to allocate boundary_needs_rank[%d]\n", rank, i);
                        for (int32_t j = 0; j < i; j++) {
                            free(boundary_needs_rank[j]);
                        }
                        free(boundary_needs_rank);
                        free(is_boundary);
                        free(local_labels);
                        free(result);
                        return NULL;
                    }
                }
                /* Mark which rank this boundary vertex depends on */
                int owner_rank = neighbor / (g_num_vertices / num_ranks);
                if (owner_rank >= num_ranks) owner_rank = num_ranks - 1;
                boundary_needs_rank[i][owner_rank] = true;
            }
        }
    }

    /* Track which ranks have completed their receives */
    bool *rank_data_ready = calloc((size_t)num_ranks, sizeof(bool));
    if (rank_data_ready == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate rank_data_ready\n", rank);
        for (int32_t i = 0; i < l_num_vertices; i++) {
            free(boundary_needs_rank[i]);
        }
        free(boundary_needs_rank);
        free(is_boundary);
        free(local_labels);
        free(result);
        return NULL;
    }

    /* Track which boundary vertices have been processed */
    bool *boundary_processed = calloc((size_t)l_num_vertices, sizeof(bool));
    if (boundary_processed == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate boundary_processed\n", rank);
        free(rank_data_ready);
        for (int32_t i = 0; i < l_num_vertices; i++) {
            free(boundary_needs_rank[i]);
        }
        free(boundary_needs_rank);
        free(is_boundary);
        free(local_labels);
        free(result);
        return NULL;
    }

    /* Label propagation with progressive async communication */
    result->num_iterations = 0;
    bool global_changed = true;

    while (global_changed) {
        result->num_iterations++;

        /* Reset tracking arrays */
        for (int r = 0; r < num_ranks; r++) {
            rank_data_ready[r] = (r == rank); /* Own rank is always ready */
        }
        for (int32_t i = 0; i < l_num_vertices; i++) {
            boundary_processed[i] = false;
        }

        /* Start asynchronous ghost exchange with separate recv/send tracking */
        MPI_Request *recv_requests = NULL;
        MPI_Request *send_requests = NULL;
        int num_recv_requests = 0;
        int num_send_requests = 0;
        int32_t **send_buffers = NULL;
        int *recv_from_rank = NULL;

        if (start_ghost_exchange_async(dg, local_labels, &recv_requests, &send_requests,
                                        &num_recv_requests, &num_send_requests,
                                        &send_buffers, &recv_from_rank) != 0) {
            fprintf(stderr, "Rank %d: Async ghost exchange failed\n", rank);
            free(boundary_processed);
            free(rank_data_ready);
            for (int32_t i = 0; i < l_num_vertices; i++) {
                free(boundary_needs_rank[i]);
            }
            free(boundary_needs_rank);
            free(is_boundary);
            free(local_labels);
            free(result);
            return NULL;
        }

        /* Phase 1: Process interior vertices (no dependencies on ghost data) */
        bool interior_changed = false;
        for (int32_t i = 0; i < l_num_vertices; i++) {
            if (is_boundary[i]) continue;

            int32_t min_label = local_labels[i];
            const int32_t edge_start = dg->local_row_ptr[i];
            const int32_t edge_end = dg->local_row_ptr[i + 1];

            for (int32_t e = edge_start; e < edge_end; e++) {
                const int32_t neighbor = dg->local_col_idx[e];
                int32_t neighbor_label = local_labels[neighbor - vertex_offset];
                if (neighbor_label < min_label) {
                    min_label = neighbor_label;
                }
            }

            if (min_label < local_labels[i]) {
                local_labels[i] = min_label;
                interior_changed = true;
            }
        }

        /* Phase 2: Progressive boundary processing as ghost data arrives */
        bool boundary_changed = false;
        int completed_recvs = 0;
        int *completed_indices = NULL;
        MPI_Status *completed_statuses = NULL;

        if (num_recv_requests > 0) {
            completed_indices = malloc(sizeof(int) * (size_t)num_recv_requests);
            completed_statuses = malloc(sizeof(MPI_Status) * (size_t)num_recv_requests);
            if (completed_indices == NULL || completed_statuses == NULL) {
                fprintf(stderr, "Rank %d: Failed to allocate completion arrays\n", rank);
                free(completed_indices);
                free(completed_statuses);
                /* Continue with fallback to Waitall */
                MPI_Waitall(num_recv_requests, recv_requests, MPI_STATUSES_IGNORE);
                for (int r = 0; r < num_ranks; r++) {
                    rank_data_ready[r] = true;
                }
            } else {
                /* Progressively check for completed receives and process ready boundary vertices */
                int remaining_recvs = num_recv_requests;
                while (remaining_recvs > 0) {
                    int outcount = 0;
                    MPI_Testsome(num_recv_requests, recv_requests, &outcount,
                                 completed_indices, completed_statuses);

                    if (outcount > 0) {
                        /* Mark ranks as ready */
                        for (int i = 0; i < outcount; i++) {
                            int idx = completed_indices[i];
                            int source_rank = recv_from_rank[idx];
                            rank_data_ready[source_rank] = true;
                        }
                        remaining_recvs -= outcount;

                        /* Process boundary vertices whose dependencies are now satisfied */
                        for (int32_t v = 0; v < l_num_vertices; v++) {
                            if (!is_boundary[v] || boundary_processed[v]) continue;

                            /* Check if all required ghost data is ready */
                            bool all_ready = true;
                            for (int r = 0; r < num_ranks; r++) {
                                if (boundary_needs_rank[v][r] && !rank_data_ready[r]) {
                                    all_ready = false;
                                    break;
                                }
                            }

                            if (all_ready) {
                                /* Process this boundary vertex */
                                int32_t min_label = local_labels[v];
                                const int32_t edge_start = dg->local_row_ptr[v];
                                const int32_t edge_end = dg->local_row_ptr[v + 1];

                                for (int32_t e = edge_start; e < edge_end; e++) {
                                    const int32_t neighbor = dg->local_col_idx[e];
                                    int32_t neighbor_label = get_vertex_label_fast(dg, local_labels, NULL, neighbor);
                                    if (neighbor_label < min_label) {
                                        min_label = neighbor_label;
                                    }
                                }

                                if (min_label < local_labels[v]) {
                                    local_labels[v] = min_label;
                                    boundary_changed = true;
                                }
                                boundary_processed[v] = true;
                            }
                        }
                    } else if (remaining_recvs > 0) {
                        /* No completions this check, do a small amount of useful work or yield */
                        /* Could process more interior vertices or other computation here */
                    }
                }
                free(completed_indices);
                free(completed_statuses);
            }
        }

        /* Process any remaining boundary vertices (shouldn't be any if logic is correct) */
        for (int32_t v = 0; v < l_num_vertices; v++) {
            if (is_boundary[v] && !boundary_processed[v]) {
                int32_t min_label = local_labels[v];
                const int32_t edge_start = dg->local_row_ptr[v];
                const int32_t edge_end = dg->local_row_ptr[v + 1];

                for (int32_t e = edge_start; e < edge_end; e++) {
                    const int32_t neighbor = dg->local_col_idx[e];
                    int32_t neighbor_label = get_vertex_label_fast(dg, local_labels, NULL, neighbor);
                    if (neighbor_label < min_label) {
                        min_label = neighbor_label;
                    }
                }

                if (min_label < local_labels[v]) {
                    local_labels[v] = min_label;
                    boundary_changed = true;
                }
            }
        }

        /* Wait for sends to complete */
        if (num_send_requests > 0) {
            MPI_Waitall(num_send_requests, send_requests, MPI_STATUSES_IGNORE);
        }

        /* Free communication resources */
        free(recv_requests);
        free(send_requests);
        free(recv_from_rank);
        if (send_buffers != NULL) {
            for (int r = 0; r < num_ranks; r++) {
                free(send_buffers[r]);
            }
            free(send_buffers);
        }

        /* Check global convergence */
        bool local_changed = interior_changed || boundary_changed;
        int local_changed_int = local_changed ? 1 : 0;
        int global_changed_int = 0;

        MPI_Allreduce(&local_changed_int, &global_changed_int, 1,
                      MPI_INT, MPI_LOR, dg->comm);

        global_changed = (global_changed_int != 0);
    }

    /* Gather final labels on all processes */
    int32_t *all_labels = NULL;
    if (rank == 0) {
        all_labels = malloc(sizeof(int32_t) * (size_t)g_num_vertices);
        if (all_labels == NULL) {
            fprintf(stderr, "Rank %d: Failed to allocate all_labels\n", rank);
            free(boundary_processed);
            free(rank_data_ready);
            for (int32_t i = 0; i < l_num_vertices; i++) {
                free(boundary_needs_rank[i]);
            }
            free(boundary_needs_rank);
            free(is_boundary);
            free(local_labels);
            free(result);
            return NULL;
        }
    }

    /* Calculate gather parameters */
    int32_t *recvcounts = NULL;
    int32_t *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(sizeof(int32_t) * (size_t)num_ranks);
        displs = malloc(sizeof(int32_t) * (size_t)num_ranks);

        const int32_t verts_per_proc = g_num_vertices / num_ranks;
        for (int r = 0; r < num_ranks; r++) {
            displs[r] = r * verts_per_proc;
            recvcounts[r] = (r == num_ranks - 1)
                            ? (g_num_vertices - displs[r])
                            : verts_per_proc;
        }
    }

    MPI_Gatherv(local_labels, l_num_vertices, MPI_INT32_T,
                all_labels, recvcounts, displs, MPI_INT32_T,
                0, dg->comm);

    /* Only rank 0 returns the full result */
    if (rank == 0) {
        result->labels = all_labels;
        result->num_components = count_unique_labels(all_labels, g_num_vertices);
        free(recvcounts);
        free(displs);
    } else {
        result->labels = NULL;
        result->num_components = 0;
    }

    /* Cleanup */
    free(boundary_processed);
    free(rank_data_ready);
    for (int32_t i = 0; i < l_num_vertices; i++) {
        free(boundary_needs_rank[i]);
    }
    free(boundary_needs_rank);
    free(is_boundary);
    free(local_labels);

    return result;
}

/* ============ MPI UNION-FIND CONNECTED COMPONENTS ============ */

/**
 * Union-Find structure for distributed algorithm
 * Stores parent pointers and ranks for local vertices only
 */
typedef struct {
    int32_t *parent;      // Parent pointers (stores GLOBAL vertex IDs)
    int32_t *rank;        // Union-by-rank
    int32_t local_n;      // Number of local vertices
    int32_t local_start;  // Global ID of first local vertex
} UnionFind;

/**
 * Update message for Union-Find
 * Sent to remote processes to propagate component information
 */
typedef struct {
    int32_t vertex;       // Global vertex ID to update
    int32_t component;    // Component root (global vertex ID)
} UFUpdate;

/**
 * Initialize Union-Find structure
 */
static int init_union_find(UnionFind *uf, int32_t local_n, int32_t local_start) {
    uf->local_n = local_n;
    uf->local_start = local_start;

    uf->parent = malloc(sizeof(int32_t) * (size_t)local_n);
    uf->rank = malloc(sizeof(int32_t) * (size_t)local_n);

    if (!uf->parent || !uf->rank) {
        free(uf->parent);
        free(uf->rank);
        return -1;
    }

    /* Initialize: each vertex is its own parent */
    for (int32_t i = 0; i < local_n; i++) {
        uf->parent[i] = local_start + i;  /* Global ID */
        uf->rank[i] = 0;
    }

    return 0;
}

/**
 * Free Union-Find structure
 */
static void free_union_find(UnionFind *uf) {
    free(uf->parent);
    free(uf->rank);
}

/**
 * Find operation with path halving
 * Path halving: make each vertex point to its grandparent during traversal
 * Achieves O(log n) amortized time with single-pass updates
 * Returns local index of root
 */
static int32_t uf_find(UnionFind *uf, int32_t x_local) {
    int32_t current = x_local;

    /* Path halving: iteratively follow parent pointers, halving path length */
    while (1) {
        int32_t parent_global = uf->parent[current];

        /* Check if parent is remote (outside local range) */
        if (parent_global < uf->local_start ||
            parent_global >= uf->local_start + uf->local_n) {
            /* Parent is remote, current is a local root pointing to remote component */
            return current;
        }

        int32_t parent_local = parent_global - uf->local_start;

        /* Check if we've reached the root */
        if (parent_local == current) {
            return current;
        }

        /* Path halving: get grandparent */
        int32_t grandparent_global = uf->parent[parent_local];

        /* Check if grandparent is remote */
        if (grandparent_global < uf->local_start ||
            grandparent_global >= uf->local_start + uf->local_n) {
            /* Grandparent is remote, parent is the local root */
            return parent_local;
        }

        /* Point current to grandparent (path halving) */
        uf->parent[current] = grandparent_global;

        /* Move to parent for next iteration */
        current = parent_local;
    }
}

/**
 * Union operation with union-by-rank and deterministic tie-breaking
 * Links larger root IDs to smaller ones for deterministic convergence
 * x_local and y_local are LOCAL indices
 * Returns 1 if union was performed, 0 otherwise
 */
static int uf_union(UnionFind *uf, int32_t x_local, int32_t y_local) {
    int32_t root_x_local = uf_find(uf, x_local);
    int32_t root_y_local = uf_find(uf, y_local);

    /* Get global IDs of roots */
    int32_t root_x_global = uf->parent[root_x_local];
    int32_t root_y_global = uf->parent[root_y_local];

    if (root_x_global == root_y_global) {
        return 0;  /* Already in same set */
    }

    /* Union by rank with deterministic tie-breaking using global IDs */
    if (uf->rank[root_x_local] < uf->rank[root_y_local]) {
        uf->parent[root_x_local] = root_y_global;
    } else if (uf->rank[root_x_local] > uf->rank[root_y_local]) {
        uf->parent[root_y_local] = root_x_global;
    } else {
        /* Equal rank: link larger ID to smaller ID for deterministic convergence */
        if (root_x_global < root_y_global) {
            uf->parent[root_y_local] = root_x_global;
        } else {
            uf->parent[root_x_local] = root_y_global;
        }
        /* Only increment rank if we modified a local root */
        if (root_x_global < root_y_global) {
            uf->rank[root_x_local]++;
        } else {
            uf->rank[root_y_local]++;
        }
    }

    return 1;
}

/**
 * Apply remote update to Union-Find structure
 * Merges local component with remote component
 */
static int apply_update(UnionFind *uf, int32_t vertex_global, int32_t component_global) {
    /* Verify vertex is local */
    if (vertex_global < uf->local_start ||
        vertex_global >= uf->local_start + uf->local_n) {
        return 0;  /* Not our vertex */
    }

    int32_t v_local = vertex_global - uf->local_start;
    int32_t root_v_local = uf_find(uf, v_local);
    int32_t root_v_global = uf->parent[root_v_local];

    if (root_v_global == component_global) {
        return 0;  /* Already in same component */
    }

    /* Merge: choose smaller global ID as canonical root */
    if (component_global < root_v_global) {
        uf->parent[root_v_local] = component_global;
        return 1;
    }

    /* If our root is smaller, remote side will update in next iteration */
    return 0;
}

/**
 * Dynamic update queue
 */
typedef struct {
    UFUpdate *data;
    int32_t size;
    int32_t capacity;
} UFUpdateQueue;

static int init_uf_queue(UFUpdateQueue *q) {
    q->capacity = 1024;
    q->size = 0;
    q->data = malloc(sizeof(UFUpdate) * (size_t)q->capacity);
    return q->data ? 0 : -1;
}

static void add_uf_update(UFUpdateQueue *q, int32_t vertex, int32_t component) {
    if (q->size >= q->capacity) {
        q->capacity *= 2;
        UFUpdate *new_data = realloc(q->data, sizeof(UFUpdate) * (size_t)q->capacity);
        if (!new_data) {
            fprintf(stderr, "Error: Failed to resize update queue\n");
            return;
        }
        q->data = new_data;
    }

    q->data[q->size].vertex = vertex;
    q->data[q->size].component = component;
    q->size++;
}

static void free_uf_queue(UFUpdateQueue *q) {
    free(q->data);
    q->data = NULL;
    q->size = 0;
    q->capacity = 0;
}

/**
 * MPI Union-Find Connected Components
 *
 * Distributed Union-Find algorithm:
 * 1. Each process owns a contiguous range of vertices
 * 2. Process local edges immediately
 * 3. Queue updates for remote edges
 * 4. Exchange updates with MPI_Alltoallv
 * 5. Apply received updates
 * 6. Repeat until convergence
 *
 * @param dg - Distributed graph structure
 * @return CCResult with component labels, or NULL on error
 */
CCResult *mpi_union_find_cc(const DistributedGraph *dg) {
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
    if (!result) {
        fprintf(stderr, "Rank %d: Failed to allocate CCResult\n", rank);
        return NULL;
    }

    /* Initialize Union-Find */
    UnionFind uf;
    if (init_union_find(&uf, l_num_vertices, vertex_offset) != 0) {
        fprintf(stderr, "Rank %d: Failed to initialize Union-Find\n", rank);
        free(result);
        return NULL;
    }

    /* Create MPI datatype for updates */
    MPI_Datatype MPI_UFUPDATE;
    MPI_Type_contiguous(2, MPI_INT32_T, &MPI_UFUPDATE);
    MPI_Type_commit(&MPI_UFUPDATE);

    /* Timing breakdown variables */
    double time_local_processing = 0.0;
    double time_mpi_alltoallv = 0.0;
    double time_other_mpi = 0.0;

    /* Main iteration loop */
    result->num_iterations = 0;
    int converged = 0;
    const int32_t verts_per_proc = g_num_vertices / num_ranks;

    while (!converged && result->num_iterations < 1000) {
        result->num_iterations++;
        int32_t local_changes = 0;
        double iter_start = MPI_Wtime();

        /* Allocate update queues (one per process) */
        UFUpdateQueue *send_queues = malloc(sizeof(UFUpdateQueue) * (size_t)num_ranks);
        if (!send_queues) {
            fprintf(stderr, "Rank %d: Failed to allocate send queues\n", rank);
            free_union_find(&uf);
            free(result);
            MPI_Type_free(&MPI_UFUPDATE);
            return NULL;
        }

        for (int p = 0; p < num_ranks; p++) {
            if (init_uf_queue(&send_queues[p]) != 0) {
                for (int j = 0; j < p; j++) {
                    free_uf_queue(&send_queues[j]);
                }
                free(send_queues);
                free_union_find(&uf);
                free(result);
                MPI_Type_free(&MPI_UFUPDATE);
                return NULL;
            }
        }

        /* PHASE 1: Process edges and build update queues */
        double phase_start = MPI_Wtime();

        for (int32_t i = 0; i < l_num_vertices; i++) {
            int32_t u_global = vertex_offset + i;
            int32_t root_u_local = uf_find(&uf, i);
            int32_t root_u_global = uf.parent[root_u_local];

            /* Process all edges of this vertex */
            for (int32_t e = dg->local_row_ptr[i]; e < dg->local_row_ptr[i + 1]; e++) {
                int32_t v_global = dg->local_col_idx[e];

                /* Skip self-loops */
                if (v_global == u_global) continue;

                /* Check if v is local or remote */
                if (v_global >= vertex_offset &&
                    v_global < vertex_offset + l_num_vertices) {
                    /* Local edge - process immediately */
                    int32_t v_local = v_global - vertex_offset;
                    if (uf_union(&uf, root_u_local, v_local)) {
                        local_changes++;
                    }
                } else {
                    /* Remote edge - queue update for owner */
                    int owner = v_global / verts_per_proc;
                    if (owner >= num_ranks) owner = num_ranks - 1;

                    if (owner >= 0 && owner < num_ranks) {
                        add_uf_update(&send_queues[owner], v_global, root_u_global);
                    }
                }
            }
        }

        time_local_processing += MPI_Wtime() - phase_start;

        /* PHASE 2: Exchange update counts */
        phase_start = MPI_Wtime();
        int32_t *send_counts = malloc(sizeof(int32_t) * (size_t)num_ranks);
        int32_t *recv_counts = malloc(sizeof(int32_t) * (size_t)num_ranks);
        int32_t *send_displs = malloc(sizeof(int32_t) * (size_t)num_ranks);
        int32_t *recv_displs = malloc(sizeof(int32_t) * (size_t)num_ranks);

        if (!send_counts || !recv_counts || !send_displs || !recv_displs) {
            fprintf(stderr, "Rank %d: Failed to allocate communication arrays\n", rank);
            for (int p = 0; p < num_ranks; p++) {
                free_uf_queue(&send_queues[p]);
            }
            free(send_queues);
            free(send_counts);
            free(recv_counts);
            free(send_displs);
            free(recv_displs);
            free_union_find(&uf);
            free(result);
            MPI_Type_free(&MPI_UFUPDATE);
            return NULL;
        }

        for (int p = 0; p < num_ranks; p++) {
            send_counts[p] = send_queues[p].size;
        }

        MPI_Alltoall(send_counts, 1, MPI_INT32_T,
                     recv_counts, 1, MPI_INT32_T, dg->comm);

        time_other_mpi += MPI_Wtime() - phase_start;

        /* Calculate displacements */
        send_displs[0] = 0;
        recv_displs[0] = 0;
        for (int p = 1; p < num_ranks; p++) {
            send_displs[p] = send_displs[p - 1] + send_counts[p - 1];
            recv_displs[p] = recv_displs[p - 1] + recv_counts[p - 1];
        }

        int32_t total_send = (num_ranks > 0) ?
            send_displs[num_ranks - 1] + send_counts[num_ranks - 1] : 0;
        int32_t total_recv = (num_ranks > 0) ?
            recv_displs[num_ranks - 1] + recv_counts[num_ranks - 1] : 0;

        /* Flatten send queues into single buffer */
        UFUpdate *send_buffer = malloc(sizeof(UFUpdate) * (size_t)(total_send > 0 ? total_send : 1));
        UFUpdate *recv_buffer = malloc(sizeof(UFUpdate) * (size_t)(total_recv > 0 ? total_recv : 1));

        if (!send_buffer || !recv_buffer) {
            fprintf(stderr, "Rank %d: Failed to allocate transfer buffers\n", rank);
            free(send_buffer);
            free(recv_buffer);
            for (int p = 0; p < num_ranks; p++) {
                free_uf_queue(&send_queues[p]);
            }
            free(send_queues);
            free(send_counts);
            free(recv_counts);
            free(send_displs);
            free(recv_displs);
            free_union_find(&uf);
            free(result);
            MPI_Type_free(&MPI_UFUPDATE);
            return NULL;
        }

        int32_t offset = 0;
        for (int p = 0; p < num_ranks; p++) {
            for (int32_t j = 0; j < send_queues[p].size; j++) {
                send_buffer[offset++] = send_queues[p].data[j];
            }
        }

        /* PHASE 3: Exchange updates */
        phase_start = MPI_Wtime();

        MPI_Alltoallv(send_buffer, send_counts, send_displs, MPI_UFUPDATE,
                      recv_buffer, recv_counts, recv_displs, MPI_UFUPDATE,
                      dg->comm);

        time_mpi_alltoallv += MPI_Wtime() - phase_start;

        /* PHASE 4: Apply received updates */
        phase_start = MPI_Wtime();
        for (int32_t i = 0; i < total_recv; i++) {
            if (apply_update(&uf, recv_buffer[i].vertex, recv_buffer[i].component)) {
                local_changes++;
            }
        }

        time_local_processing += MPI_Wtime() - phase_start;

        /* Cleanup iteration resources */
        for (int p = 0; p < num_ranks; p++) {
            free_uf_queue(&send_queues[p]);
        }
        free(send_queues);
        free(send_counts);
        free(recv_counts);
        free(send_displs);
        free(recv_displs);
        free(send_buffer);
        free(recv_buffer);

        /* PHASE 5: Check convergence */
        phase_start = MPI_Wtime();

        int32_t global_changes = 0;
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_INT32_T,
                      MPI_SUM, dg->comm);

        time_other_mpi += MPI_Wtime() - phase_start;

        converged = (global_changes == 0);

        /* Progress logging */
        if (rank == 0 && (result->num_iterations % 10 == 0 || converged)) {
            printf("UF Iteration %d: %d changes\n",
                   result->num_iterations, global_changes);
            fflush(stdout);
        }
    }

    if (rank == 0 && result->num_iterations >= 1000) {
        printf("Warning: Union-Find reached max iterations (1000)\n");
        fflush(stdout);
    }

    /* Final path compression */
    for (int32_t i = 0; i < l_num_vertices; i++) {
        uf_find(&uf, i);
    }

    /* Gather all labels to rank 0 */
    int32_t *all_labels = NULL;
    if (rank == 0) {
        all_labels = malloc(sizeof(int32_t) * (size_t)g_num_vertices);
        if (!all_labels) {
            fprintf(stderr, "Rank %d: Failed to allocate all_labels\n", rank);
            free_union_find(&uf);
            free(result);
            MPI_Type_free(&MPI_UFUPDATE);
            return NULL;
        }
    }

    /* Prepare gather parameters */
    int32_t *recvcounts = NULL;
    int32_t *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(sizeof(int32_t) * (size_t)num_ranks);
        displs = malloc(sizeof(int32_t) * (size_t)num_ranks);

        if (!recvcounts || !displs) {
            fprintf(stderr, "Rank %d: Failed to allocate gather arrays\n", rank);
            free(all_labels);
            free(recvcounts);
            free(displs);
            free_union_find(&uf);
            free(result);
            MPI_Type_free(&MPI_UFUPDATE);
            return NULL;
        }

        for (int r = 0; r < num_ranks; r++) {
            displs[r] = r * verts_per_proc;
            recvcounts[r] = (r == num_ranks - 1)
                            ? (g_num_vertices - displs[r])
                            : verts_per_proc;
        }
    }

    MPI_Gatherv(uf.parent, l_num_vertices, MPI_INT32_T,
                all_labels, recvcounts, displs, MPI_INT32_T,
                0, dg->comm);

    /* Count components on rank 0 */
    if (rank == 0) {
        result->labels = all_labels;
        result->num_components = count_unique_labels(all_labels, g_num_vertices);
        free(recvcounts);
        free(displs);

        /* Print timing breakdown (matching report section 4.5) */
        double total_time = time_local_processing + time_mpi_alltoallv + time_other_mpi;
        if (total_time > 0.001) {  /* Only print for non-trivial runs */
            printf("\n=== UF Communication Breakdown ===\n");
            printf("%-30s %10.5f s (%5.1f%%)\n",
                   "Local Edge Processing",
                   time_local_processing,
                   100.0 * time_local_processing / total_time);
            printf("%-30s %10.5f s (%5.1f%%)\n",
                   "MPI_Alltoallv (updates)",
                   time_mpi_alltoallv,
                   100.0 * time_mpi_alltoallv / total_time);
            printf("%-30s %10.5f s (%5.1f%%)\n",
                   "Other MPI Operations",
                   time_other_mpi,
                   100.0 * time_other_mpi / total_time);
            printf("%-30s %10.5f s (%5.1f%%)\n",
                   "Total",
                   total_time,
                   100.0);
            fflush(stdout);
        }
    } else {
        result->labels = NULL;
        result->num_components = 0;
    }

    /* Cleanup */
    free_union_find(&uf);
    MPI_Type_free(&MPI_UFUPDATE);

    return result;
}


