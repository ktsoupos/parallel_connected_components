#pragma once

#include "graph.h"
#include "cc_sequential.h"

/**
 * OpenMP hello world test - prints from each thread
 */
void openmp_hello_world(void);

/**
 * Parallel connected components using synchronous label propagation (OpenMP)
 * Each iteration: all vertices update labels in parallel, then synchronize
 * Uses double buffering to avoid race conditions
 */
CCResult* label_propagation_sync_omp(const Graph* restrict g, int num_threads);

/**
 * Parallel connected components using asynchronous label propagation (OpenMP)
 * Updates happen in-place with atomic operations
 * Faster convergence and better performance than synchronous version
 */
CCResult* label_propagation_async_omp(const Graph* restrict g, int num_threads);

/**
 * Set number of OpenMP threads (for testing)
 * Returns the number actually set
 */
int set_omp_threads(int num_threads);

/**
 * Get current number of OpenMP threads
 */
int get_omp_threads(void);
