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

/**
 * Sample frequent element from component array
 * Returns the most frequently occurring component ID from random samples
 * Used in Afforest to identify the largest component
 */
int32_t sample_frequent_element(const int32_t *comp, int32_t num_vertices, int32_t num_samples);