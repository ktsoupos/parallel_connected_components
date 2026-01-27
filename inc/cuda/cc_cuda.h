#pragma once

#include "cc_sequential.h"
#include "graph.h"

#ifdef __cplusplus
extern "C" {

#endif

/**
 * CUDA Connected Components - GPU-accelerated algorithms
 *
 * This module implements parallel connected components algorithms on GPU
 * using NVIDIA CUDA for high-performance graph processing.
 */

/**
 * Check if CUDA is available and print device information
 * Returns 0 on success, -1 if no CUDA devices found
 */
int cuda_check_device(void);

/**
 * Helper function: allocate graph data on GPU device memory
 * Copies CSR arrays from host to device
 *
 * @param g Input graph (host memory)
 * @param d_row_ptr Device pointer for row pointers (output)
 * @param d_col_idx Device pointer for column indices (output)
 * @return 0 on success, -1 on error
 */
int cuda_allocate_graph(const Graph *restrict g, int32_t **d_row_ptr, int32_t **d_col_idx);

/**
 * Helper function: free GPU device memory
 */
void cuda_free_graph(int32_t *d_row_ptr, int32_t *d_col_idx);

/**
 * CUDA Label Propagation Connected Components
 * Uses synchronous label propagation on GPU
 * @param g Input graph
 * @return CCResult containing labels, component count, and iteration count
 */
CCResult *cc_cuda(const Graph *restrict g);

/**
 * ECL-CC: High-Performance Connected Components for GPUs
 * Based on Jaiganesh & Burtscher, HPDC 2018
 *
 * Uses degree-based work distribution:
 * - Low-degree vertices: thread granularity
 * - Medium-degree vertices: warp granularity
 * - High-degree vertices: block granularity
 *
 * @param g Input graph (CSR format)
 * @return CCResult containing labels, component count, and iteration count
 */
CCResult *cc_cuda_ecl(const Graph *restrict g);

/**
 * Afforest: Sampling-based Connected Components for GPUs
 * Based on Sutton, Ben-Nun, Hoefler, IPDPS 2018
 *
 * Two-phase algorithm:
 * - Phase 1: Sample first k neighbors per vertex (quick connectivity)
 * - Phase 2: Process remaining edges, skipping large components
 *
 * Best for graphs where sampling quickly connects most vertices
 * (e.g., social networks, power-law graphs)
 *
 * @param g Input graph (CSR format)
 * @return CCResult containing labels, component count, and iteration count
 */
CCResult *cc_cuda_afforest(const Graph *restrict g);

/* Count unique components from labels */
int32_t count_components(const int32_t *labels, int32_t n);

/**
 * Destroy CCResult allocated by CUDA functions
 * Properly handles pinned memory allocated by cudaHostAlloc
 *
 * @param result CCResult to destroy (may have pinned memory)
 */
void cc_cuda_result_destroy(CCResult *result);

#ifdef __cplusplus
}
#endif