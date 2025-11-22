#pragma once

#include "graph.h"
#include "cc_sequential.h"
#include <stdint.h>

/**
 * Synchronous label propagation using simple pthreads (SIMPLIFIED VERSION)
 * Each iteration, all vertices update their label to the minimum neighbor label
 * Iterates until no changes occur (convergence) or max iterations reached
 *
 * Uses simple pthread create/join pattern (no work-stealing complexity)
 * Static scheduling with cache-aligned thread arguments
 *
 * Returns: CCResult with labels and iteration count, or NULL on error
 */
CCResult* label_propagation_sync_pthreads(const Graph* g, int32_t num_threads);

/**
 * Asynchronous label propagation using work-stealing threadpool
 * When a vertex label changes, tasks are created for affected neighbors
 * More dynamic than sync version, can converge faster on some graphs
 *
 * WARNING: Can create many tasks on dense graphs - use sync version for large/dense graphs
 *
 * Returns: CCResult with labels (num_iterations = 0 for async), or NULL on error
 */
CCResult* label_propagation_async_pthreads(const Graph* g, int32_t num_threads);

/**
 * Afforest - Lock-free parallel connected components (FASTEST)
 * Uses atomic union-find with path compression
 * Alternates between link phase (connect to neighbors) and compress phase
 * Typically converges in 2-3 iterations, much faster than label propagation
 *
 * @param neighbor_rounds: Number of sampling rounds (default 2 if <= 0)
 * Returns: CCResult with labels and iteration count, or NULL on error
 */
CCResult* afforest_pthreads(const Graph* g, int32_t num_threads, int32_t neighbor_rounds);

