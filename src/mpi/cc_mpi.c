#include "cc_mpi.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

/**
 * Add a remote edge to the appropriate buffer, resizing if necessary
 */
static int add_remote_edge(RemoteBuffer *buf, int32_t local_u, int32_t global_v) {
    // Check if buffer is full
    if (buf->count >= buf->capacity) {
        // Double the capacity
        int32_t new_capacity = buf->capacity * 2;

        int32_t *new_local = realloc(buf->local_vertices, sizeof(int32_t) * (size_t)new_capacity);
        int32_t *new_global = realloc(buf->global_vertices, sizeof(int32_t) * (size_t)new_capacity);
        int32_t *new_parents = realloc(buf->parent_values, sizeof(int32_t) * (size_t)new_capacity);

        if (new_local == NULL || new_global == NULL || new_parents == NULL) {
            fprintf(stderr, "Error: Failed to resize remote buffer\n");
            free(new_local);
            free(new_global);
            free(new_parents);
            return -1;
        }

        buf->local_vertices = new_local;
        buf->global_vertices = new_global;
        buf->parent_values = new_parents;
        buf->capacity = new_capacity;
    }

    // Add the edge
    buf->local_vertices[buf->count] = local_u;
    buf->global_vertices[buf->count] = global_v;
    buf->count++;

    return 0;
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

int partition_graph(const Graph *global_graph, DistributedGraph **dist_graph, MPI_Comm comm) {
    if (dist_graph == NULL) {
        fprintf(stderr, "Error: NULL dist_graph pointer\n");
        return -1;
    }

    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    // Only rank 0 should have a valid global_graph
    if (rank == 0 && global_graph == NULL) {
        fprintf(stderr, "Error: NULL global_graph pointer on rank 0\n");
        return -1;
    }

    int32_t metadata[2] = {0, 0};
    if (rank == 0) {
        metadata[0] = global_graph->num_vertices;
        metadata[1] = global_graph->num_edges;

    }
    MPI_Bcast(metadata, 2, MPI_INT32_T, 0, comm);
    const int32_t g_num_vertices = metadata[0];
    const int32_t g_num_edges = metadata[1];

    const int32_t verts_per_proc = g_num_vertices / num_ranks;
    const int32_t vertex_offset = rank * verts_per_proc;
    int32_t l_num_vertices;

    if (rank == num_ranks - 1) {
        // Last rank takes remaining vertices
        l_num_vertices = g_num_vertices - vertex_offset;
    } else {
        l_num_vertices = verts_per_proc;
    }

    *dist_graph = malloc(sizeof(DistributedGraph));
    if (*dist_graph == NULL) {
        fprintf(stderr, "Error: Failed to allocate DistributedGraph\n");
        return -1;
    }
    (*dist_graph)->g_num_vertices = g_num_vertices;
    (*dist_graph)->g_num_edges = g_num_edges;
    (*dist_graph)->l_num_vertices = l_num_vertices;
    (*dist_graph)->vertex_offset = vertex_offset;
    (*dist_graph)->rank = rank;
    (*dist_graph)->num_ranks = num_ranks;
    (*dist_graph)->comm = comm;

    (*dist_graph)->local_row_ptr = malloc(sizeof(int32_t) * (size_t)(l_num_vertices + 1));
    if ((*dist_graph)->local_row_ptr == NULL) {
        fprintf(stderr, "Error: Failed to allocate local row ptr\n");
        free(*dist_graph);
        return -1;
    }

    int32_t *sendcounts = malloc(sizeof(int32_t) * (size_t)num_ranks);
    if (sendcounts == NULL) {
        fprintf(stderr, "Error: Failed to allocate sendcounts\n");
        free((*dist_graph)->local_row_ptr);
        free(*dist_graph);
        return -1;
    }

    int32_t *displs = malloc(sizeof(int32_t) * (size_t)num_ranks);
    if (displs == NULL) {
        fprintf(stderr, "Error: Failed to allocate displs\n");
        free(sendcounts);
        free((*dist_graph)->local_row_ptr);
        free(*dist_graph);
        return -1;
    }

    if (rank == 0) {
        for (int i = 0; i < num_ranks; i++) {
            const int32_t offset = i * verts_per_proc;
            const int32_t count = (i == num_ranks - 1)
                                  ? (g_num_vertices - offset + 1)
                                  : (verts_per_proc + 1);
            sendcounts[i] = count;
            displs[i] = offset;
        }
    }
    // clang-format off
    MPI_Scatterv(
        (rank == 0) ? global_graph->row_ptr : NULL,  // sendbuf (only rank 0)
        sendcounts,                                // send counts per process
        displs,                                    // displacements
        MPI_INT32_T,                               // sendtype
        (*dist_graph)->local_row_ptr,              // recvbuf
        l_num_vertices + 1,                        // recvcount
        MPI_INT32_T,                               // recvtype
        0,                                         // root
        comm
    );
    // clang-format on

    //The row_ptr values need to be adjusted to start from 0:
    const int32_t offset_adjustment = (*dist_graph)->local_row_ptr[0];
    for (int32_t i = 0; i <= l_num_vertices; i++) {
        (*dist_graph)->local_row_ptr[i] -= offset_adjustment;
    }

    (*dist_graph)->l_num_edges = (*dist_graph)->local_row_ptr[l_num_vertices];

    if (rank == 0) {
        printf("Rank 0: Allocating %d edges for distribution\n", (*dist_graph)->l_num_edges);
    }

    (*dist_graph)->local_col_idx = malloc(sizeof(int32_t) * (size_t)(*dist_graph)->l_num_edges);
    if ((*dist_graph)->local_col_idx == NULL) {
        fprintf(stderr, "Rank %d: Failed to allocate local col idx (%d edges)\n",
                rank, (*dist_graph)->l_num_edges);
        free((*dist_graph)->local_row_ptr);
        free(*dist_graph);
        free(sendcounts);
        free(displs);
        return -1;
    }

    // On rank 0: calculate edge counts and offsets for each process
    if (rank == 0) {
        for (int i = 0; i < num_ranks; i++) {
            const int32_t v_offset = i * verts_per_proc;
            const int32_t v_count = (i == num_ranks - 1)
                                    ? (g_num_vertices - v_offset)
                                    : verts_per_proc;

            const int32_t edge_start = global_graph->row_ptr[v_offset];
            const int32_t edge_end = global_graph->row_ptr[v_offset + v_count];

            sendcounts[i] = edge_end - edge_start;
            displs[i] = edge_start;
        }
    }

    MPI_Scatterv(
        rank == 0 ? global_graph->col_idx : NULL,
        sendcounts,
        displs,
        MPI_INT32_T,
        (*dist_graph)->local_col_idx,
        (*dist_graph)->l_num_edges,
        MPI_INT32_T,
        0,
        comm
        );

    // Cleanup temporary arrays
    free(sendcounts);
    free(displs);

    return 0;
}

/**
 * Link two vertices u and v using union-find with path compression
 * Based on the Link function from the GAP Benchmark Suite Afforest implementation
 */
__attribute__((always_inline)) inline static void link_vertices(const int32_t u, const int32_t v,
                                                                int32_t *restrict parents) {
    /* Read parent values */
    int32_t p1 = parents[u];
    int32_t p2 = parents[v];

    while (p1 != p2) {
        const int32_t high = (p1 > p2) ? p1 : p2;
        const int32_t low = (p1 < p2) ? p1 : p2;
        const int32_t p_high = parents[high];
        int32_t expected = high;

        if ((p_high == low) || // Was already 'low'
            (p_high == high &&
             (__atomic_compare_exchange_n( // Succeeded on writing 'low'
                 &parents[high], &expected, low, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)))) {
            break;
        }

        p1 = parents[expected]; // Update with actual value after CAS
        p2 = parents[low];
    }
}

__attribute__((always_inline)) inline static int get_owner_rank(
    int32_t global_vertex_id, const DistributedGraph *dg) {
    int32_t verts_per_proc = dg->g_num_vertices / dg->num_ranks;
    int owner = global_vertex_id / verts_per_proc;

    // Handle last rank which may have more vertices
    if (owner >= dg->num_ranks) {
        owner = dg->num_ranks - 1;
    }

    return owner;
}

static void link_with_remote_parent(int32_t u_local, int32_t remote_parent_global,
                                    int32_t *parents) {
    int32_t u_parent = parents[u_local];

    while (u_parent != remote_parent_global) {
        const int32_t high = (u_parent > remote_parent_global) ? u_parent : remote_parent_global;
        const int32_t low = (u_parent < remote_parent_global) ? u_parent : remote_parent_global;

        // If u's parent is the higher one, try to update it
        if (u_parent == high) {
            int32_t expected = u_parent;
            if (__atomic_compare_exchange_n(&parents[u_local], &expected, low,
                                            false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
                break; // Successfully updated
            }
            u_parent = expected; // CAS failed, retry with new value
        } else {
            // Remote parent is higher, nothing to update locally
            break;
        }
    }
}

static void exchange_and_link_remote(RemoteBuffer *buffers, int32_t *parents,
                                     const DistributedGraph *dg) {
    CommData *comm = prepare_comm_data(buffers, dg->num_ranks, dg->comm);
    if (comm == NULL) {
        fprintf(stderr, "Error: Failed to prepare communication data\n");
        return;
    }

    // Early exit if no communication needed
    if (comm->total_send == 0 && comm->total_recv == 0) {
        free(comm->send_counts);
        comm->send_counts = NULL;
        free(comm->recv_counts);
        comm->recv_counts = NULL;
        free(comm->send_displs);
        comm->send_displs = NULL;
        free(comm->recv_displs);
        comm->recv_displs = NULL;
        free(comm);
        return;
    }

    int32_t *send_buf = malloc(sizeof(int32_t) * (size_t)comm->total_send);
    if (send_buf == NULL) {
        fprintf(stderr, "Error: Failed to allocate send buffer\n");
        free(comm->send_counts);
        comm->send_counts = NULL;
        free(comm->recv_counts);
        comm->recv_counts = NULL;
        free(comm->send_displs);
        comm->send_displs = NULL;
        free(comm->recv_displs);
        comm->recv_displs = NULL;
        free(comm);
        return;
    }

    int idx = 0;
    for (int rank = 0; rank < dg->num_ranks; rank++) {
        for (int i = 0; i < buffers[rank].count; i++) {
            send_buf[idx++] = buffers[rank].global_vertices[i];
        }
    }

    int32_t *recv_buf = malloc(sizeof(int32_t) * (size_t)comm->total_recv);
    if (recv_buf == NULL) {
        fprintf(stderr, "Error: Failed to allocate recv buffer\n");
        free(send_buf);
        free(comm->send_counts);
        comm->send_counts = NULL;
        free(comm->recv_counts);
        comm->recv_counts = NULL;
        free(comm->send_displs);
        comm->send_displs = NULL;
        free(comm->recv_displs);
        comm->recv_displs = NULL;
        free(comm);
        return;
    }
    MPI_Alltoallv(
        send_buf, comm->send_counts, comm->send_displs, MPI_INT32_T,
        recv_buf, comm->recv_counts, comm->recv_displs, MPI_INT32_T,
        dg->comm
        );

    int32_t *response_buf = malloc(sizeof(int32_t) * (size_t)comm->total_recv);
    if (response_buf == NULL) {
        fprintf(stderr, "Error: Failed to allocate response buffer\n");
        free(send_buf);
        free(recv_buf);
        free(comm->send_counts);
        comm->send_counts = NULL;
        free(comm->recv_counts);
        comm->recv_counts = NULL;
        free(comm->send_displs);
        comm->send_displs = NULL;
        free(comm->recv_displs);
        comm->recv_displs = NULL;
        free(comm);
        return;
    }

    // For each requested vertex, lookup its parent value
    for (int i = 0; i < comm->total_recv; i++) {
        const int32_t global_id = recv_buf[i];

        // Check if this vertex belongs to us
        if (global_id >= dg->vertex_offset &&
            global_id < dg->vertex_offset + dg->l_num_vertices) {
            const int32_t local_id = global_id - dg->vertex_offset;
            response_buf[i] = parents[local_id];
        } else {
            // This shouldn't happen, but return the global ID as fallback
            fprintf(stderr, "Warning: Rank %d received invalid request for vertex %d (owns %d-%d)\n",
                    dg->rank, global_id, dg->vertex_offset,
                    dg->vertex_offset + dg->l_num_vertices - 1);
            response_buf[i] = global_id;
        }
    }
    int32_t *parent_recv_buf = malloc(sizeof(int32_t) * (size_t)comm->total_send);
    if (parent_recv_buf == NULL) {
        fprintf(stderr, "Error: Failed to allocate parent receive buffer\n");
        free(send_buf);
        free(recv_buf);
        free(response_buf);
        free(comm->send_counts);
        comm->send_counts = NULL;
        free(comm->recv_counts);
        comm->recv_counts = NULL;
        free(comm->send_displs);
        comm->send_displs = NULL;
        free(comm->recv_displs);
        comm->recv_displs = NULL;
        free(comm);
        return;
    }

    // Exchange: Send parent values back, receive parent values you requested
    // Note: send and recv are swapped
    MPI_Alltoallv(
        response_buf, comm->recv_counts, comm->recv_displs, MPI_INT32_T,
        parent_recv_buf, comm->send_counts, comm->send_displs, MPI_INT32_T,
        dg->comm
        );

    idx = 0;
    for (int rank = 0; rank < dg->num_ranks; rank++) {
        for (int i = 0; i < buffers[rank].count; i++) {
            const int32_t u_local = buffers[rank].local_vertices[i];
            const int32_t parent_v = parent_recv_buf[idx++];

            // Link u with the remote parent value
            link_with_remote_parent(u_local, parent_v, parents);
        }
    }

    free(send_buf);
    free(recv_buf);
    free(response_buf);
    free(parent_recv_buf);
    free(comm->send_counts);
    comm->send_counts = NULL;
    free(comm->recv_counts);
    comm->recv_counts = NULL;
    free(comm->send_displs);
    comm->send_displs = NULL;
    free(comm->recv_displs);
    comm->recv_displs = NULL;
    free(comm);
}


static void compress_local(int32_t *parents, int32_t num_vertices,
                           int32_t vertex_offset, int32_t l_num_vertices) {
    for (int32_t n = 0; n < num_vertices; n++) {
        const int32_t parent_global = parents[n];
        // Check if parent is a local vertex
        if (parent_global >= vertex_offset &&
            parent_global < vertex_offset + l_num_vertices) {

            // Parent is local - convert to local index
            const int32_t parent_local = parent_global - vertex_offset;
            const int32_t grandparent_global = parents[parent_local];

            // Compress: point directly to grandparent
            if (grandparent_global != parent_global) {
                parents[n] = grandparent_global;
            }
        }
        // If parent is remote, we can't compress without communication
    }
}


CCResult *afforest_mpi(const DistributedGraph *dist_graph, int32_t neighbor_rounds) {
    if (dist_graph == NULL) {
        fprintf(stderr, "Error: DistributedGraph is NULL\n");
        return NULL;
    }

    int32_t *parents = malloc(sizeof(int32_t) * (size_t)dist_graph->l_num_vertices);
    if (parents == NULL) {
        fprintf(stderr, "Error: malloc failed for parents array\n");
        return NULL;
    }
    for (int32_t i = 0; i < dist_graph->l_num_vertices; i++) {
        parents[i] = i + dist_graph->vertex_offset;
    }

    if (neighbor_rounds <= 0) {
        neighbor_rounds = 2; // fine-tuned
    }

    // Allocate remote buffers for cross-partition edges
    const int32_t initial_capacity = estimate_remote_capacity(dist_graph, neighbor_rounds);
    RemoteBuffer *remote_bufs = alloc_remote_buffers(dist_graph->num_ranks, initial_capacity);
    if (remote_bufs == NULL) {
        fprintf(stderr, "Error: Failed to allocate remote buffers\n");
        free(parents);
        return NULL;
    }

    // Neighbor sampling rounds - collect local and remote edges
    for (int32_t r = 0; r < neighbor_rounds; r++) {
        for (int32_t u = 0; u < dist_graph->l_num_vertices; u++) {
            const int32_t start = dist_graph->local_row_ptr[u];
            const int32_t end = dist_graph->local_row_ptr[u + 1];
            const int32_t num_neighbors = end - start;

            if (r < num_neighbors) {
                const int32_t v = dist_graph->local_col_idx[start + r]; // get the r-th neighbor
                const bool is_local = (v >= dist_graph->vertex_offset) &&
                                      (v < dist_graph->vertex_offset + dist_graph->l_num_vertices);

                if (is_local) {
                    // Local edge - link immediately
                    const int32_t v_local = v - dist_graph->vertex_offset;
                    // Bounds check before linking
                    if (u >= 0 && u < dist_graph->l_num_vertices &&
                        v_local >= 0 && v_local < dist_graph->l_num_vertices) {
                        link_vertices(u, v_local, parents);
                    }
                } else {
                    // Remote edge - buffer for later exchange
                    const int owner = get_owner_rank(v, dist_graph);
                    add_remote_edge(&remote_bufs[owner], u, v);
                }
            }
        }
    }

    // Exchange and link remote edges (batched communication)
    exchange_and_link_remote(remote_bufs, parents, dist_graph);

    // Cleanup remote buffers
    for (int i = 0; i < dist_graph->num_ranks; i++) {
        free(remote_bufs[i].local_vertices);
        remote_bufs[i].local_vertices = NULL;
        free(remote_bufs[i].global_vertices);
        remote_bufs[i].global_vertices = NULL;
        free(remote_bufs[i].parent_values);
        remote_bufs[i].parent_values = NULL;
    }
    free(remote_bufs);

    // Compress parent pointers (one-hop local compression)
    compress_local(parents, dist_graph->l_num_vertices,
                   dist_graph->vertex_offset, dist_graph->l_num_vertices);

    // Phase 3: Sample to identify largest component
    // NOTE: Disabled to save memory - allocating g_num_vertices on each process is wasteful
    #define NUM_SAMPLES 0
    int32_t largest_component = -1;  // Skip largest component optimization

    #if 0  // Disabled memory-intensive sampling
    int32_t *sample_counts = calloc((size_t)dist_graph->g_num_vertices, sizeof(int32_t));
    if (sample_counts == NULL) {
        fprintf(stderr, "Error: Failed to allocate sample counts\n");
        free(parents);
        return NULL;
    }

    // Sample with replacement
    unsigned int seed = (unsigned int)(dist_graph->rank + time(NULL));
    for (int32_t i = 0; i < NUM_SAMPLES; i++) {
        int32_t idx = rand_r(&seed) % dist_graph->l_num_vertices;
        int32_t component_id = parents[idx];

        if (component_id >= 0 && component_id < dist_graph->g_num_vertices) {
            sample_counts[component_id]++;
        }
    }

    // Global reduction to sum sample counts
    int32_t *global_sample_counts = malloc(sizeof(int32_t) * (size_t)dist_graph->g_num_vertices);
    if (global_sample_counts == NULL) {
        fprintf(stderr, "Error: Failed to allocate global sample counts\n");
        free(sample_counts);
        free(parents);
        return NULL;
    }

    MPI_Allreduce(sample_counts, global_sample_counts, dist_graph->g_num_vertices,
                  MPI_INT32_T, MPI_SUM, dist_graph->comm);
    free(sample_counts);

    // Find the largest component
    int32_t largest_component = 0;
    int32_t max_count = 0;

    for (int32_t i = 0; i < dist_graph->g_num_vertices; i++) {
        if (global_sample_counts[i] > max_count) {
            max_count = global_sample_counts[i];
            largest_component = i;
        }
    }
    free(global_sample_counts);

    if (dist_graph->rank == 0 && max_count > 0) {
        float percentage = (float)max_count / ((float)NUM_SAMPLES * (float)dist_graph->num_ranks) * 100.0f;
        printf("Skipping largest component (ID: %d, %.1f%% of samples)\n",
               largest_component, percentage);
    }
    #endif  // End of disabled sampling code

    // Phase 4: Final linking phase - process remaining neighbors
    RemoteBuffer *final_remote_bufs = alloc_remote_buffers(dist_graph->num_ranks, initial_capacity);
    if (final_remote_bufs == NULL) {
        fprintf(stderr, "Error: Failed to allocate final remote buffers\n");
        free(parents);
        return NULL;
    }

    for (int32_t u = 0; u < dist_graph->l_num_vertices; u++) {
        // Skip vertices already in largest component
        if (parents[u] == largest_component) {
            continue;
        }

        const int32_t start = dist_graph->local_row_ptr[u];
        const int32_t end = dist_graph->local_row_ptr[u + 1];
        const int32_t num_neighbors = end - start;

        // Process remaining neighbors (after neighbor_rounds)
        for (int32_t j = neighbor_rounds; j < num_neighbors; j++) {
            const int32_t v = dist_graph->local_col_idx[start + j];
            const bool is_local = (v >= dist_graph->vertex_offset) &&
                                  (v < dist_graph->vertex_offset + dist_graph->l_num_vertices);

            if (is_local) {
                const int32_t v_local = v - dist_graph->vertex_offset;
                // Bounds check before linking
                if (u >= 0 && u < dist_graph->l_num_vertices &&
                    v_local >= 0 && v_local < dist_graph->l_num_vertices) {
                    link_vertices(u, v_local, parents);
                }
            } else {
                const int owner = get_owner_rank(v, dist_graph);
                add_remote_edge(&final_remote_bufs[owner], u, v);
            }
        }
    }

    // Exchange and link final remote edges
    exchange_and_link_remote(final_remote_bufs, parents, dist_graph);
    for (int i = 0; i < dist_graph->num_ranks; i++) {
        free(final_remote_bufs[i].local_vertices);
        final_remote_bufs[i].local_vertices = NULL;
        free(final_remote_bufs[i].global_vertices);
        final_remote_bufs[i].global_vertices = NULL;
        free(final_remote_bufs[i].parent_values);
        final_remote_bufs[i].parent_values = NULL;
    }
    free(final_remote_bufs);

    // Phase 5: Final compression
    compress_local(parents, dist_graph->l_num_vertices,
                   dist_graph->vertex_offset, dist_graph->l_num_vertices);

    // Phase 6: Count components - gather all parents to rank 0
    int32_t *all_parents = NULL;
    int32_t *recvcounts = NULL;
    int32_t *displs = NULL;

    if (dist_graph->rank == 0) {
        all_parents = malloc(sizeof(int32_t) * (size_t)dist_graph->g_num_vertices);
        recvcounts = malloc(sizeof(int32_t) * (size_t)dist_graph->num_ranks);
        displs = malloc(sizeof(int32_t) * (size_t)dist_graph->num_ranks);

        if (all_parents == NULL || recvcounts == NULL || displs == NULL) {
            fprintf(stderr, "Error: Failed to allocate gathering buffers\n");
            free(all_parents);
            free(recvcounts);
            free(displs);
            free(parents);
            return NULL;
        }

        int32_t verts_per_proc = dist_graph->g_num_vertices / dist_graph->num_ranks;
        for (int i = 0; i < dist_graph->num_ranks; i++) {
            int offset = i * verts_per_proc;
            recvcounts[i] = (i == dist_graph->num_ranks - 1)
                ? (dist_graph->g_num_vertices - offset)
                : verts_per_proc;
            displs[i] = offset;
        }
    }

    MPI_Gatherv(parents, dist_graph->l_num_vertices, MPI_INT32_T,
                all_parents, recvcounts, displs, MPI_INT32_T,
                0, dist_graph->comm);

    int32_t num_components = 0;
    if (dist_graph->rank == 0) {
        // Count unique component IDs
        bool *seen = calloc((size_t)dist_graph->g_num_vertices, sizeof(bool));
        if (seen == NULL) {
            fprintf(stderr, "Error: Failed to allocate seen array\n");
            free(all_parents);
            free(recvcounts);
            free(displs);
            free(parents);
            return NULL;
        }

        for (int32_t i = 0; i < dist_graph->g_num_vertices; i++) {
            int32_t root = all_parents[i];
            if (root >= 0 && root < dist_graph->g_num_vertices && !seen[root]) {
                seen[root] = true;
                num_components++;
            }
        }

        free(seen);
        free(all_parents);
        free(recvcounts);
        free(displs);

        printf("Found %d connected components\n", num_components);
    }

    // Broadcast component count to all processes
    MPI_Bcast(&num_components, 1, MPI_INT32_T, 0, dist_graph->comm);

    // Build and return result
    CCResult *result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        free(parents);
        return NULL;
    }

    result->labels = parents;
    result->num_components = num_components;
    result->num_iterations = neighbor_rounds + 1;

    return result;
}

/* ============ SHILOACH-VISHKIN ALGORITHM ============ */

/**
 * Find root of vertex with path compression
 */
static int32_t find_root(int32_t v_local, int32_t *parents) {
    int32_t root = parents[v_local];
    if (root == parents[v_local])
        return root;

    // Path compression
    int32_t current = v_local;
    while (parents[current] != root) {
        int32_t next = parents[current];
        parents[current] = root;
        current = next;
        root = parents[current];
    }
    return root;
}

/**
 * Hybrid Shiloach-Vishkin: Local union-find + distributed hooking
 */
CCResult *shiloach_vishkin_mpi(const DistributedGraph *dist_graph) {
    if (dist_graph == NULL) {
        fprintf(stderr, "Error: DistributedGraph is NULL\n");
        return NULL;
    }

    int32_t *parents = malloc(sizeof(int32_t) * (size_t)dist_graph->l_num_vertices);
    if (parents == NULL) {
        fprintf(stderr, "Error: malloc failed for parents array\n");
        return NULL;
    }

    // Initialize: each vertex is its own parent (using global IDs)
    for (int32_t i = 0; i < dist_graph->l_num_vertices; i++) {
        parents[i] = i + dist_graph->vertex_offset;
    }

    /* ===== PHASE 1: Local Union-Find (no communication) ===== */
    for (int32_t u = 0; u < dist_graph->l_num_vertices; u++) {
        const int32_t start = dist_graph->local_row_ptr[u];
        const int32_t end = dist_graph->local_row_ptr[u + 1];

        for (int32_t j = start; j < end; j++) {
            const int32_t v_global = dist_graph->local_col_idx[j];

            // Only process local edges
            if (v_global >= dist_graph->vertex_offset &&
                v_global < dist_graph->vertex_offset + dist_graph->l_num_vertices) {
                const int32_t v_local = v_global - dist_graph->vertex_offset;
                // Bounds check
                if (v_local >= 0 && v_local < dist_graph->l_num_vertices) {
                    link_vertices(u, v_local, parents);
                }
            }
        }
    }

    /* ===== PHASE 2: Shiloach-Vishkin for boundary vertices ===== */

    // Allocate boundary exchange buffers
    int32_t *send_buf = malloc(sizeof(int32_t) * (size_t)dist_graph->l_num_vertices);
    int32_t *recv_buf = malloc(sizeof(int32_t) * (size_t)dist_graph->g_num_vertices);

    if (send_buf == NULL || recv_buf == NULL) {
        fprintf(stderr, "Error: Failed to allocate exchange buffers\n");
        free(send_buf);
        free(recv_buf);
        free(parents);
        return NULL;
    }

    // Initialize recv_buf with all parent values before starting iterations
    int32_t *recvcounts_init = malloc(sizeof(int32_t) * (size_t)dist_graph->num_ranks);
    int32_t *displs_init = malloc(sizeof(int32_t) * (size_t)dist_graph->num_ranks);

    if (recvcounts_init == NULL || displs_init == NULL) {
        fprintf(stderr, "Error: Failed to allocate initial gather buffers\n");
        free(recvcounts_init);
        free(displs_init);
        free(send_buf);
        free(recv_buf);
        free(parents);
        return NULL;
    }

    int32_t verts_per_proc = dist_graph->g_num_vertices / dist_graph->num_ranks;
    for (int i = 0; i < dist_graph->num_ranks; i++) {
        int offset = i * verts_per_proc;
        recvcounts_init[i] = (i == dist_graph->num_ranks - 1)
            ? (dist_graph->g_num_vertices - offset)
            : verts_per_proc;
        displs_init[i] = offset;
    }

    for (int32_t i = 0; i < dist_graph->l_num_vertices; i++) {
        send_buf[i] = parents[i];
    }

    MPI_Allgatherv(send_buf, dist_graph->l_num_vertices, MPI_INT32_T,
                   recv_buf, recvcounts_init, displs_init, MPI_INT32_T,
                   dist_graph->comm);

    free(recvcounts_init);
    recvcounts_init = NULL;
    free(displs_init);
    displs_init = NULL;

    int32_t num_iterations = 0;
    const int32_t MAX_ITERATIONS = 100;
    bool global_converged = false;

    while (!global_converged && num_iterations < MAX_ITERATIONS) {
        num_iterations++;
        bool local_changed = false;

        /* Hooking phase: link to minimum neighbor */
        for (int32_t u = 0; u < dist_graph->l_num_vertices; u++) {
            int32_t my_parent = parents[u];
            int32_t min_neighbor = my_parent;

            const int32_t start = dist_graph->local_row_ptr[u];
            const int32_t end = dist_graph->local_row_ptr[u + 1];

            // Find minimum among all neighbors
            for (int32_t j = start; j < end; j++) {
                const int32_t v_global = dist_graph->local_col_idx[j];

                if (v_global >= dist_graph->vertex_offset &&
                    v_global < dist_graph->vertex_offset + dist_graph->l_num_vertices) {
                    // Local neighbor - direct access
                    const int32_t v_local = v_global - dist_graph->vertex_offset;
                    const int32_t neighbor_parent = parents[v_local];
                    if (neighbor_parent < min_neighbor) {
                        min_neighbor = neighbor_parent;
                    }
                } else {
                    // Remote neighbor - use last known value from recv_buf
                    if (v_global >= 0 && v_global < dist_graph->g_num_vertices) {
                        const int32_t neighbor_parent = recv_buf[v_global];
                        if (neighbor_parent < min_neighbor) {
                            min_neighbor = neighbor_parent;
                        }
                    }
                }
            }

            // Hook to minimum
            if (min_neighbor < my_parent) {
                parents[u] = min_neighbor;
                local_changed = true;
            }
        }

        /* Pointer jumping: compress paths */
        for (int32_t u = 0; u < dist_graph->l_num_vertices; u++) {
            int32_t parent = parents[u];

            // Check if parent is local
            if (parent >= dist_graph->vertex_offset &&
                parent < dist_graph->vertex_offset + dist_graph->l_num_vertices) {
                int32_t parent_local = parent - dist_graph->vertex_offset;
                int32_t grandparent = parents[parent_local];

                if (grandparent < parent) {
                    parents[u] = grandparent;
                    local_changed = true;
                }
            } else {
                // Parent is remote - use recv_buf
                if (parent >= 0 && parent < dist_graph->g_num_vertices) {
                    int32_t grandparent = recv_buf[parent];
                    if (grandparent < parent) {
                        parents[u] = grandparent;
                        local_changed = true;
                    }
                }
            }
        }

        /* Exchange parent values via Allgatherv */
        int32_t *recvcounts = NULL;
        int32_t *displs = NULL;

        if (dist_graph->rank == 0) {
            recvcounts = malloc(sizeof(int32_t) * (size_t)dist_graph->num_ranks);
            displs = malloc(sizeof(int32_t) * (size_t)dist_graph->num_ranks);

            if (recvcounts == NULL || displs == NULL) {
                free(recvcounts);
                free(displs);
                free(send_buf);
                free(recv_buf);
                free(parents);
                return NULL;
            }

            int32_t verts_per_proc = dist_graph->g_num_vertices / dist_graph->num_ranks;
            for (int i = 0; i < dist_graph->num_ranks; i++) {
                int offset = i * verts_per_proc;
                recvcounts[i] = (i == dist_graph->num_ranks - 1)
                    ? (dist_graph->g_num_vertices - offset)
                    : verts_per_proc;
                displs[i] = offset;
            }
        }

        // Broadcast recvcounts and displs to all ranks
        if (dist_graph->rank != 0) {
            recvcounts = malloc(sizeof(int32_t) * (size_t)dist_graph->num_ranks);
            displs = malloc(sizeof(int32_t) * (size_t)dist_graph->num_ranks);
        }

        MPI_Bcast(recvcounts, dist_graph->num_ranks, MPI_INT32_T, 0, dist_graph->comm);
        MPI_Bcast(displs, dist_graph->num_ranks, MPI_INT32_T, 0, dist_graph->comm);

        // Copy local parents to send buffer
        for (int32_t i = 0; i < dist_graph->l_num_vertices; i++) {
            send_buf[i] = parents[i];
        }

        MPI_Allgatherv(send_buf, dist_graph->l_num_vertices, MPI_INT32_T,
                       recv_buf, recvcounts, displs, MPI_INT32_T,
                       dist_graph->comm);

        free(recvcounts);
        recvcounts = NULL;
        free(displs);
        displs = NULL;

        /* Check global convergence */
        MPI_Allreduce(&local_changed, &global_converged, 1, MPI_C_BOOL,
                      MPI_LOR, dist_graph->comm);
        global_converged = !global_converged;
    }

    free(send_buf);
    send_buf = NULL;
    free(recv_buf);
    recv_buf = NULL;

    if (dist_graph->rank == 0) {
        printf("Shiloach-Vishkin converged in %d iterations\n", num_iterations);
    }

    /* ===== PHASE 3: Count components ===== */
    int32_t *all_parents = NULL;
    int32_t *recvcounts = NULL;
    int32_t *displs = NULL;

    if (dist_graph->rank == 0) {
        all_parents = malloc(sizeof(int32_t) * (size_t)dist_graph->g_num_vertices);
        recvcounts = malloc(sizeof(int32_t) * (size_t)dist_graph->num_ranks);
        displs = malloc(sizeof(int32_t) * (size_t)dist_graph->num_ranks);

        if (all_parents == NULL || recvcounts == NULL || displs == NULL) {
            fprintf(stderr, "Error: Failed to allocate gathering buffers\n");
            free(all_parents);
            free(recvcounts);
            free(displs);
            free(parents);
            return NULL;
        }

        int32_t verts_per_proc = dist_graph->g_num_vertices / dist_graph->num_ranks;
        for (int i = 0; i < dist_graph->num_ranks; i++) {
            int offset = i * verts_per_proc;
            recvcounts[i] = (i == dist_graph->num_ranks - 1)
                ? (dist_graph->g_num_vertices - offset)
                : verts_per_proc;
            displs[i] = offset;
        }
    }

    MPI_Gatherv(parents, dist_graph->l_num_vertices, MPI_INT32_T,
                all_parents, recvcounts, displs, MPI_INT32_T,
                0, dist_graph->comm);

    int32_t num_components = 0;
    if (dist_graph->rank == 0) {
        bool *seen = calloc((size_t)dist_graph->g_num_vertices, sizeof(bool));
        if (seen == NULL) {
            fprintf(stderr, "Error: Failed to allocate seen array\n");
            free(all_parents);
            free(recvcounts);
            free(displs);
            free(parents);
            return NULL;
        }

        for (int32_t i = 0; i < dist_graph->g_num_vertices; i++) {
            int32_t root = all_parents[i];
            if (root >= 0 && root < dist_graph->g_num_vertices && !seen[root]) {
                seen[root] = true;
                num_components++;
            }
        }

        free(seen);
        free(all_parents);
        free(recvcounts);
        free(displs);

        printf("Found %d connected components\n", num_components);
    }

    MPI_Bcast(&num_components, 1, MPI_INT32_T, 0, dist_graph->comm);

    CCResult *result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        free(parents);
        return NULL;
    }

    result->labels = parents;
    result->num_components = num_components;
    result->num_iterations = num_iterations;

    return result;
}