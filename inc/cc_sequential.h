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
 */
CCResult* label_propagation_min(const Graph* g);

/**
 * Free CCResult memory
 */
void cc_result_destroy(CCResult* result);

/**
 * Print connected components statistics
 */
void cc_result_print_stats(const CCResult* result, const Graph* g);