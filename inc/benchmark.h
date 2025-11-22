#pragma once

#include "graph.h"

/**
 * Run all sequential connected components algorithms and compare performance
 * Returns 0 on success, non-zero on error
 */
int run_sequential_benchmarks(const Graph* g);

#ifdef _OPENMP
/**
 * Run all parallel connected components algorithms and compare with sequential
 * Returns 0 on success, non-zero on error
 */
int run_parallel_benchmarks(const Graph* g, int num_threads);
#endif

/**
 * Run pthreads work-stealing async label propagation benchmark
 * Returns 0 on success, non-zero on error
 */
int run_pthreads_benchmarks(const Graph* g, int num_threads);
