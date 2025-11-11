#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/*
 * Graph stored in Compressed Sparse Row (CSR) format
 * Efficient for sparse graphs and parallel processing
 */
typedef struct {
    int32_t num_vertices;
    int32_t num_edges;

    /* CSR format (valid after finalization) */
    int32_t* row_ptr;      /* Size: num_vertices + 1 */
    int32_t* col_idx;      /* Size: 2 * num_edges (for undirected) */

    /* Temporary storage during construction */
    int32_t** adj_lists;   /* Adjacency lists per vertex */
    int32_t* degrees;      /* Current number of neighbors per vertex */
    int32_t* capacities;   /* Allocated capacity for each adj_list */

    bool finalized;        /* false = construction phase, true = finalized (CSR ready) */
} Graph;

/*
 * Create a new graph with n vertices
 * Returns: pointer to Graph on success, NULL on failure
 */
Graph* graph_create(int32_t num_vertices);

/*
 * Add an undirected edge between vertices u and v
 * Can only be called before graph_finalize()
 * Returns: 0 on success, -1 on error
 */
int32_t graph_add_edge(Graph* g, int32_t u, int32_t v);

/*
 * Finalize the graph: convert temporary adjacency lists to CSR format
 * Must be called after all edges are added, before using the graph
 * Returns: 0 on success, -1 on error
 */
int32_t graph_finalize(Graph* g);

/*
 * Free all memory associated with the graph
 */
void graph_destroy(Graph* g);

/*
 * Get the number of vertices
 * Returns: number of vertices, or -1 on error
 */
int32_t graph_get_num_vertices(const Graph* g);

/*
 * Get the number of edges
 * Returns: number of edges, or -1 on error
 */
int32_t graph_get_num_edges(const Graph* g);

/*
 * Get the neighbors of vertex v in CSR format
 * Returns: pointer to neighbor array on success, NULL on error
 * Sets *num_neighbors to the count (only valid after graph_finalize())
 */
const int32_t* graph_get_neighbors(const Graph* g, int32_t v, int32_t* num_neighbors);

/*
 * Print graph statistics
 */
void graph_print_stats(const Graph* g);
