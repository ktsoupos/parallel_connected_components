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


CCResult *afforest_mpi(const DistributedGraph *dist_graph, int32_t neighbor_rounds);

/**
 * Hybrid Shiloach-Vishkin Algorithm with Local Union-Find
 *
 * Three-phase approach optimized for distributed systems:
 * 1. Local union-find within partition (no communication)
 * 2. Shiloach-Vishkin hooking/jumping for boundary vertices
 * 3. Global label propagation to finalize components
 *
 * @param dist_graph - Distributed graph structure
 * @return CCResult with component labels and statistics
 */
CCResult* shiloach_vishkin_mpi(const DistributedGraph *dist_graph);
