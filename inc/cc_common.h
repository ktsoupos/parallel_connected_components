#pragma once

#include <stdint.h>
#include <stddef.h>

/**
 * Count number of unique labels (connected components)
 * Returns: number of unique components, or -1 on error
 */
int32_t count_unique_labels(const int32_t *labels, int32_t num_vertices);

/**
 * Print component size statistics
 */
void print_component_stats(const int32_t *labels, int32_t num_vertices);
