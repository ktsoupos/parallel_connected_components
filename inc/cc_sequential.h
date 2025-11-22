#pragma once

#include "graph.h"

typedef struct {
    int32_t* labels;           // Label for each vertex
    int32_t num_components;    // Total components found
    int32_t num_iterations;    // Iterations to converge
} CCResult;

/**
 * Run sequential connected components using label propagation
 * Optimized with queue + bool array hybrid approach
 * Uses restrict keyword for better compiler optimization
 */
CCResult* label_propagation_min(const Graph* restrict g);

/**
 * Simple baseline label propagation (no optimizations)
 * Processes all vertices each iteration for comparison
 */
CCResult* label_propagation_min_simple(const Graph* restrict g);

/**
 * Union-Find connected components (optimal sequential algorithm)
 * Uses path compression and union by minimum
 * Time: O(m * α(n)) where α is inverse Ackermann (nearly O(m))
 */
CCResult* union_find_cc(const Graph* restrict g);

/**
 * Union-Find with edge reordering optimization
 * Only processes each edge once (u < v), better cache locality
 */
CCResult* union_find_cc_edge_reorder(const Graph* restrict g);

/**
 * Free CCResult memory
 */
void cc_result_destroy(CCResult* result);

/**
 * Print connected components statistics
 */
void cc_result_print_stats(const CCResult* result, const Graph* g);
