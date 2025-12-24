#pragma once

#include "graph.h"

/**
 * Run OpenCilk parallel benchmarks
 *
 * @param g Graph to analyze
 * @param num_workers Number of Cilk workers to use
 * @return 0 on success, -1 on failure
 */
int run_opencilk_benchmarks(const Graph *g, int num_workers);
