#include "cc_cuda.h"
#include "cc_common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/* CUDA error checking macro */
#define CUDA_CHECK(call)                                                                               \
    do {                                                                                              \
        cudaError_t err = call;                                                                       \
        if (err != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                        \
                    cudaGetErrorString(err));                                                         \
            return NULL;                                                                              \
        }                                                                                             \
    } while (0)

#define CUDA_CHECK_RETURN(call, retval)                                                               \
    do {                                                                                              \
        cudaError_t err = call;                                                                       \
        if (err != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                        \
                    cudaGetErrorString(err));                                                         \
            return retval;                                                                            \
        }                                                                                             \
    } while (0)

/* CUDA kernel configuration */
#define BLOCK_SIZE 256
#define WARP_SIZE 32


int cuda_check_device(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }

    printf("\n=== CUDA Device Information ===\n");
    printf("Number of CUDA devices: %d\n\n", device_count);

    for (int dev = 0; dev < device_count; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n", (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("\n");
    }

    return 0;
}

int cuda_allocate_graph(const Graph *restrict g, int32_t **d_row_ptr, int32_t **d_col_idx) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return -1;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);
    const int32_t num_edges = graph_get_num_edges(g);

    /* Allocate device memory for CSR arrays */
    /* Note: col_idx stores both directions for undirected graphs, so size is 2 * num_edges */
    CUDA_CHECK_RETURN(cudaMalloc((void**)d_row_ptr, sizeof(int32_t) * (num_vertices + 1)), -1);
    CUDA_CHECK_RETURN(cudaMalloc((void**)d_col_idx, sizeof(int32_t) * 2 * num_edges), -1);

    /* Copy CSR data from host to device */
    const int32_t *h_row_ptr = g->row_ptr;
    const int32_t *h_col_idx = g->col_idx;

    CUDA_CHECK_RETURN(cudaMemcpy(*d_row_ptr, h_row_ptr, sizeof(int32_t) * (num_vertices + 1),
                                 cudaMemcpyHostToDevice), -1);
    CUDA_CHECK_RETURN(cudaMemcpy(*d_col_idx, h_col_idx, sizeof(int32_t) * 2 * num_edges,
                                 cudaMemcpyHostToDevice), -1);

    return 0;
}

void cuda_free_graph(int32_t *d_row_ptr, int32_t *d_col_idx) {
    if (d_row_ptr) cudaFree(d_row_ptr);
    if (d_col_idx) cudaFree(d_col_idx);
}


/* Initialize labels: label[v] = v */
__global__ void init_labels_kernel(int32_t *labels, int32_t n) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        labels[v] = v;
    }
}

/* Label propagation kernel: each vertex gets minimum label of neighbors */
__global__ void propagate_kernel(int32_t *labels,
                                  const int32_t *row_ptr,
                                  const int32_t *col_idx,
                                  int32_t n,
                                  int32_t *changed) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    int32_t my_label = labels[v];
    int32_t new_label = my_label;

    /* Check all neighbors */
    int32_t start = row_ptr[v];
    int32_t end = row_ptr[v + 1];

    for (int32_t e = start; e < end; e++) {
        int32_t neighbor = col_idx[e];
        int32_t neighbor_label = labels[neighbor];
        if (neighbor_label < new_label) {
            new_label = neighbor_label;
        }
    }

    /* Update if changed */
    if (new_label < my_label) {
        labels[v] = new_label;
        atomicOr(changed, 1);
    }
}




/* Pointer jumping kernel: compress label paths */
__global__ void compress_labels_kernel(int32_t *labels, int32_t n, int32_t *changed) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    int32_t label = labels[v];
    int32_t parent = labels[label];

    /* Path compression: point directly to root */
    if (parent != label) {
        labels[v] = parent;
        atomicOr(changed, 1);
    }
}

CCResult *cc_cuda(const Graph *restrict g) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);
    const int32_t num_edges = graph_get_num_edges(g);

    if (num_vertices <= 0) {
        fprintf(stderr, "Error: Invalid number of vertices\n");
        return NULL;
    }

    printf("Running CUDA label propagation on %d vertices, %d edges\n", num_vertices, num_edges);

    /* Allocate result structure */
    CCResult *result = (CCResult *)malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        return NULL;
    }

    /* Allocate host labels */
    result->labels = (int32_t *)malloc(sizeof(int32_t) * num_vertices);
    if (result->labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate labels array\n");
        free(result);
        return NULL;
    }

    /* Allocate graph on device */
    int32_t *d_row_ptr = NULL;
    int32_t *d_col_idx = NULL;
    if (cuda_allocate_graph(g, &d_row_ptr, &d_col_idx) != 0) {
        fprintf(stderr, "Error: Failed to allocate graph on device\n");
        free(result->labels);
        free(result);
        return NULL;
    }

    /* Allocate device labels and changed flag */
    int32_t *d_labels = NULL;
    int32_t *d_changed = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_labels, sizeof(int32_t) * num_vertices));
    CUDA_CHECK(cudaMalloc((void **)&d_changed, sizeof(int32_t)));

    /* Configure kernel launch parameters */
    int32_t num_blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Initialize labels on device */
    init_labels_kernel<<<num_blocks, BLOCK_SIZE>>>(d_labels, num_vertices);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Label propagation iterations */
    result->num_iterations = 0;
    int32_t h_changed = 1;
    const int32_t MAX_ITERATIONS = 1000;

    while (h_changed && result->num_iterations < MAX_ITERATIONS) {
        result->num_iterations++;
        h_changed = 0;

        /* Reset changed flag on device */
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int32_t), cudaMemcpyHostToDevice));

        /* Run propagation kernel */
        propagate_kernel<<<num_blocks, BLOCK_SIZE>>>(d_labels, d_row_ptr, d_col_idx,
                                                      num_vertices, d_changed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Check if any changes occurred */
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost));
    }

    printf("Label propagation converged in %d iterations\n", result->num_iterations);

    /* Compress labels (pointer jumping to find roots) */
    int32_t compress_iterations = 0;
    h_changed = 1;
    while (h_changed && compress_iterations < 100) {
        compress_iterations++;
        h_changed = 0;

        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int32_t), cudaMemcpyHostToDevice));

        compress_labels_kernel<<<num_blocks, BLOCK_SIZE>>>(d_labels, num_vertices, d_changed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost));
    }

    printf("Label compression completed in %d iterations\n", compress_iterations);

    /* Copy labels back to host */
    CUDA_CHECK(cudaMemcpy(result->labels, d_labels, sizeof(int32_t) * num_vertices,
                          cudaMemcpyDeviceToHost));

    /* Count components */
    result->num_components = count_components(result->labels, num_vertices);
    printf("Found %d connected components\n", result->num_components);

    /* Cleanup device memory */
    cudaFree(d_labels);
    cudaFree(d_changed);
    cuda_free_graph(d_row_ptr, d_col_idx);

    return result;
}

int32_t count_components(const int32_t *labels, int32_t n) {
    int32_t count = 0;
    for (int32_t i = 0; i < n; i++) {
        if (labels[i] == i) {
            count++;
        }
    }
    return count;
}







