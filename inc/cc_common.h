#pragma once

#include <stddef.h>
#include <stdint.h>

/**
 * Make restrict keyword work with both C and C++ (CUDA)
 * In C++/CUDA, restrict is not standard, so we use __restrict__
 */
#ifdef __cplusplus
    #define restrict __restrict__
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Count number of unique labels (connected components)
 * Returns: number of unique components, or -1 on error
 */
int32_t count_unique_labels(const int32_t *labels, int32_t num_vertices);

/**
 * Print component size statistics
 */
void print_component_stats(const int32_t *labels, int32_t num_vertices);

#ifdef __cplusplus
}
#endif
