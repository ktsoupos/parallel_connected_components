#pragma once

#include "graph.h"
#include "cc_sequential.h"
#include <stdint.h>

/**
 * Async label propagation using work-stealing thread pool
 * Each vertex asynchronously updates its label to the minimum of its neighbors
 * When a label changes, tasks are created for affected neighbors
 *
 * NOTE: For large/dense graphs, use label_propagation_sync_pthreads instead
 * Returns: CCResult with labels (num_iterations = 0 for async), or NULL on error
 */
CCResult* label_propagation_async_pthreads(const Graph* g, int32_t num_threads);

/**
 * Synchronous batched label propagation (RECOMMENDED for large graphs)
 * Processes vertices in parallel chunks with barriers between iterations
 * More memory-efficient than async version (no task explosion)
 *
 * Returns: CCResult with labels and iteration count, or NULL on error
 */
CCResult* label_propagation_sync_pthreads(const Graph* g, int32_t num_threads);

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

