#pragma once

#include "graph.h"
#include "cc_sequential.h"
#include <stdint.h>

/**
 * Afforest Algorithm - Simple Pthreads Implementation
 *
 * A simplified version of the Afforest algorithm using basic pthreads
 * instead of a work-stealing thread pool. Uses atomic operations for
 * lock-free union-find with path compression.
 *
 * Algorithm phases:
 * 1. Neighbor sampling rounds - Process first k neighbors in parallel
 * 2. Path compression after each round
 * 3. Identify largest component via sampling
 * 4. Final linking phase - Process remaining neighbors (skipping largest component)
 * 5. Final path compression
 *
 * @param g: Input graph in CSR format
 * @param num_threads: Number of worker threads to use
 * @param neighbor_rounds: Number of initial neighbor sampling rounds (default 2 if <= 0)
 * @return CCResult with component labels and statistics, or NULL on error
 */
CCResult* afforest_simple_pthreads(const Graph* g, int32_t num_threads, int32_t neighbor_rounds);
