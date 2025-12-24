#pragma once

#include "graph.h"

/**
 * Read a graph from a Matrix Market (.mtx) file
 *
 * Supports coordinate format, pattern or real values, symmetric matrices
 * Expected format:
 *   %%MatrixMarket matrix coordinate pattern symmetric
 *   % comments...
 *   num_rows num_cols num_entries
 *   row1 col1 [value1]
 *   row2 col2 [value2]
 *   ...
 *
 * Returns: Graph pointer on success, NULL on failure
 */
Graph *read_mtx_file(const char *filename);

/**
 * Read MTX file with progress reporting
 * Reports progress every 'report_interval' edges
 *
 * Returns: Graph pointer on success, NULL on failure
 */
Graph *read_mtx_file_verbose(const char *filename, int32_t report_interval);
