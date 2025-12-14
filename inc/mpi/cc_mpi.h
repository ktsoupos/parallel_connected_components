#pragma once
#include "cc_sequential.h"

#include <mpi.h>
#include <stdint.h>

#include "graph.h"

typedef struct {
    int32_t g_num_vertices;
    int32_t g_num_edges;
    int32_t l_num_vertices;
    int32_t l_num_edges;
    int32_t vertex_offset;
    int32_t *local_row_ptr; // Size: l_num_vertices + 1
    int32_t *local_col_idx; // Size: l_num_edges,  stores GLOBAL IDs
    int rank;
    int num_ranks;
    MPI_Comm comm;

    /* Ghost vertex communication data */
    int32_t num_ghost_vertices;        // Number of ghost vertices (owned by others, needed locally)
    int32_t *ghost_global_ids;         // Global IDs of ghost vertices
    int32_t *ghost_to_owner;           // Which rank owns each ghost vertex
    int32_t *ghost_labels;             // Labels for ghost vertices

    /* Per-rank communication metadata */
    int32_t *send_counts;              // Number of vertices to send to each rank
    int32_t *recv_counts;              // Number of vertices to receive from each rank
    int32_t *send_displs;              // Displacement for send buffer per rank
    int32_t *recv_displs;              // Displacement for recv buffer per rank
    int32_t **send_vertices;           // Local indices to send to each rank

    char buff[8];
}  __attribute__((aligned(64))) DistributedGraph;

/**
 * Partition and distribute graph across processes
 *
 * @param global_graph - Complete graph (only valid on rank 0, NULL on others)
 * @param dist_graph - Output: allocated DistributedGraph
 * @param comm - MPI communicator
 * @return 0 on success, non-zero on error
 */
int partition_graph(const Graph *global_graph, DistributedGraph **dist_graph, MPI_Comm comm);

/**
 * MPI Label Propagation for Connected Components
 *
 * @param dg - Distributed graph structure
 * @return CCResult with component labels (rank 0 only), or NULL on error
 */
CCResult *mpi_label_propagation(const DistributedGraph *dg);

/**
 * Optimized MPI Label Propagation with Ghost/Halo Exchange
 *
 * Uses:
 * - Ghost vertex communication (only boundary vertices)
 * - Asynchronous MPI (non-blocking send/recv)
 * - Computation/communication overlap
 *
 * @param dg - Distributed graph structure with ghost metadata
 * @return CCResult with component labels (rank 0 only), or NULL on error
 */
CCResult *mpi_label_propagation_optimized(const DistributedGraph *dg);