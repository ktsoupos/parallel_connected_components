#include "cc_mpi.h"

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

/* ============ REMOTE BUFFER MANAGEMENT ============ */

/**
 * Estimate initial capacity for remote buffers based on graph structure
 */
static int32_t estimate_remote_capacity(const DistributedGraph *dg, int32_t neighbor_rounds) {
    if (dg->l_num_vertices == 0)
        return 256;

    // Average degree per vertex
    int32_t avg_degree = dg->l_num_edges / dg->l_num_vertices;

    // Estimate: (avg_degree * neighbor_rounds * probability_remote)
    // probability_remote ≈ (num_ranks - 1) / num_ranks
    int32_t estimated = (avg_degree * neighbor_rounds * (dg->num_ranks - 1)) / dg->num_ranks;

    // Per-rank estimate (edges distributed across all ranks)
    estimated = estimated / dg->num_ranks;

    // Safety margin (2x) and bounds
    estimated *= 2;
    if (estimated < 256)
        estimated = 256;
    if (estimated > 65536)
        estimated = 65536;

    return estimated;
}

/**
 * Allocate array of RemoteBuffers (one per MPI rank)
 */
static RemoteBuffer *alloc_remote_buffers(int num_ranks, int32_t initial_capacity) {
    RemoteBuffer *buffers = calloc((size_t)num_ranks, sizeof(RemoteBuffer));
    if (buffers == NULL) {
        fprintf(stderr, "Error: Failed to allocate remote buffers\n");
        return NULL;
    }

    for (int i = 0; i < num_ranks; i++) {
        buffers[i].capacity = initial_capacity;
        buffers[i].count = 0;
        buffers[i].local_vertices = malloc(sizeof(int32_t) * (size_t)initial_capacity);
        buffers[i].global_vertices = malloc(sizeof(int32_t) * (size_t)initial_capacity);
        buffers[i].parent_values = malloc(sizeof(int32_t) * (size_t)initial_capacity);

        if (buffers[i].local_vertices == NULL ||
            buffers[i].global_vertices == NULL ||
            buffers[i].parent_values == NULL) {
            fprintf(stderr, "Error: Failed to allocate buffer arrays\n");
            for (int j = 0; j < num_ranks; j++) {
                free(buffers[j].local_vertices);
                buffers[j].local_vertices = NULL;
                free(buffers[j].global_vertices);
                buffers[j].global_vertices = NULL;
                free(buffers[j].parent_values);
                buffers[j].parent_values = NULL;
            }
            free(buffers);
            return NULL;
        }
    }

    return buffers;
}


/* ============ COMMUNICATION DATA MANAGEMENT ============ */

/**
 * Allocate CommData structure
 */
static CommData *alloc_comm_data(int num_ranks) {
    CommData *comm = malloc(sizeof(CommData));
    if (comm == NULL) {
        fprintf(stderr, "Error: Failed to allocate CommData\n");
        return NULL;
    }

    comm->send_counts = calloc((size_t)num_ranks, sizeof(int32_t));
    comm->recv_counts = calloc((size_t)num_ranks, sizeof(int32_t));
    comm->send_displs = calloc((size_t)num_ranks, sizeof(int32_t));
    comm->recv_displs = calloc((size_t)num_ranks, sizeof(int32_t));

    if (comm->send_counts == NULL || comm->recv_counts == NULL ||
        comm->send_displs == NULL || comm->recv_displs == NULL) {
        fprintf(stderr, "Error: Failed to allocate CommData arrays\n");
        free(comm->send_counts);
        comm->send_counts = NULL;
        free(comm->recv_counts);
        comm->recv_counts = NULL;
        free(comm->send_displs);
        comm->send_displs = NULL;
        free(comm->recv_displs);
        comm->recv_displs = NULL;
        free(comm);
        return NULL;
    }

    comm->total_send = 0;
    comm->total_recv = 0;

    return comm;
}

/**
 * Prepare communication metadata from remote buffers
 */
static CommData *prepare_comm_data(RemoteBuffer *buffers, int num_ranks, MPI_Comm mpi_comm) {
    CommData *comm = alloc_comm_data(num_ranks);
    if (comm == NULL)
        return NULL;

    // Fill send counts from buffer counts
    for (int i = 0; i < num_ranks; i++) {
        comm->send_counts[i] = buffers[i].count;
    }

    // Exchange counts with all processes
    MPI_Alltoall(comm->send_counts, 1, MPI_INT32_T,
                 comm->recv_counts, 1, MPI_INT32_T, mpi_comm);

    // Calculate displacements and totals
    comm->send_displs[0] = 0;
    comm->recv_displs[0] = 0;

    for (int i = 1; i < num_ranks; i++) {
        comm->send_displs[i] = comm->send_displs[i - 1] + comm->send_counts[i - 1];
        comm->recv_displs[i] = comm->recv_displs[i - 1] + comm->recv_counts[i - 1];
    }

    comm->total_send = comm->send_displs[num_ranks - 1] + comm->send_counts[num_ranks - 1];
    comm->total_recv = comm->recv_displs[num_ranks - 1] + comm->recv_counts[num_ranks - 1];

    return comm;
}

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


