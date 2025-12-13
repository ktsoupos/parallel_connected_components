#pragma once
#include "cc_sequential.h"
#include <stdint.h>

/**
 * Sample frequent element from component array
 * Returns the most frequently occurring component ID from random samples
 * Used in Afforest to identify the largest component
 */
int32_t sample_frequent_element_cilk(const int32_t *comp, int32_t num_vertices,
                                     int32_t num_samples);

/**
 * Afforest algorithm - Parallel connected components using OpenCilk
 * Based on the algorithm from the GAP Benchmark Suite
 */
CCResult *afforest_cilk(const Graph *restrict g, int num_threads, int32_t neighbor_rounds);

/**
 * Recursive Edge-Based Union-Find - Connected Components
 * Showcases divide-and-conquer parallelism with heavy cilk_spawn usage
 *
 * @param g Graph to process
 * @param num_threads Number of workers (ignored - use CILK_NWORKERS)
 * @return CCResult containing component labels
 */
CCResult *recursive_edge_cc(const Graph *restrict g, int num_threads);
