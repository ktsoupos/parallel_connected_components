#include "cc_cuda.h"
#include "cc_common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

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

/* ECL-CC constants */
#define ECL_THREADS_PER_BLOCK 256
#define ECL_WARPSIZE 32

/* ECL-CC device variables for worklist management */
static __device__ int ecl_topL, ecl_posL, ecl_topH, ecl_posH;

/* Helper to get wall time */
static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}


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

/* ========================================================================
 * ECL-CC Algorithm Kernels
 * Based on: Jaiganesh & Burtscher, "A High-Performance Connected Components
 * Implementation for GPUs", HPDC 2018
 * ======================================================================== */

/* ECL-CC: Initialize with first smaller neighbor ID */
static __global__ __launch_bounds__(ECL_THREADS_PER_BLOCK, 2048 / ECL_THREADS_PER_BLOCK)
void ecl_init(const int nodes, const int32_t* const __restrict__ nidx,
              const int32_t* const __restrict__ nlist, int32_t* const __restrict__ nstat)
{
    const int from = threadIdx.x + blockIdx.x * ECL_THREADS_PER_BLOCK;
    const int incr = gridDim.x * ECL_THREADS_PER_BLOCK;

    for (int v = from; v < nodes; v += incr) {
        const int beg = nidx[v];
        const int end = nidx[v + 1];
        int m = v;
        int i = beg;
        while ((m == v) && (i < end)) {
            m = min(m, nlist[i]);
            i++;
        }
        nstat[v] = m;
    }

    if (from == 0) {
        ecl_topL = 0;
        ecl_posL = 0;
        ecl_topH = nodes - 1;
        ecl_posH = nodes - 1;
    }
}

/* ECL-CC: Intermediate pointer jumping */
static inline __device__ int ecl_representative(const int idx, int32_t* const __restrict__ nstat)
{
    int curr = nstat[idx];
    if (curr != idx) {
        int next, prev = idx;
        while (curr > (next = nstat[curr])) {
            nstat[prev] = next;
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

/* ECL-CC: Process low-degree vertices at thread granularity and fill worklists */
static __global__ __launch_bounds__(ECL_THREADS_PER_BLOCK, 2048 / ECL_THREADS_PER_BLOCK)
void ecl_compute1(const int nodes, const int32_t* const __restrict__ nidx,
                  const int32_t* const __restrict__ nlist, int32_t* const __restrict__ nstat,
                  int* const __restrict__ wl)
{
    const int from = threadIdx.x + blockIdx.x * ECL_THREADS_PER_BLOCK;
    const int incr = gridDim.x * ECL_THREADS_PER_BLOCK;

    for (int v = from; v < nodes; v += incr) {
        const int vstat = nstat[v];
        if (v != vstat) {
            const int beg = nidx[v];
            const int end = nidx[v + 1];
            int deg = end - beg;
            if (deg > 16) {
                int idx;
                if (deg <= 352) {
                    idx = atomicAdd(&ecl_topL, 1);
                } else {
                    idx = atomicAdd(&ecl_topH, -1);
                }
                wl[idx] = v;
            } else {
                int vstat = ecl_representative(v, nstat);
                for (int i = beg; i < end; i++) {
                    const int nli = nlist[i];
                    if (v > nli) {
                        int ostat = ecl_representative(nli, nstat);
                        bool repeat;
                        do {
                            repeat = false;
                            if (vstat != ostat) {
                                int ret;
                                if (vstat < ostat) {
                                    if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                                        ostat = ret;
                                        repeat = true;
                                    }
                                } else {
                                    if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                                        vstat = ret;
                                        repeat = true;
                                    }
                                }
                            }
                        } while (repeat);
                    }
                }
            }
        }
    }
}

/* ECL-CC: Process medium-degree vertices at warp granularity */
static __global__ __launch_bounds__(ECL_THREADS_PER_BLOCK, 2048 / ECL_THREADS_PER_BLOCK)
void ecl_compute2(const int nodes, const int32_t* const __restrict__ nidx,
                  const int32_t* const __restrict__ nlist, int32_t* const __restrict__ nstat,
                  const int* const __restrict__ wl)
{
    const int lane = threadIdx.x % ECL_WARPSIZE;

    int idx;
    if (lane == 0) idx = atomicAdd(&ecl_posL, 1);
    idx = __shfl_sync(0xffffffff, idx, 0);
    while (idx < ecl_topL) {
        const int v = wl[idx];
        int vstat = ecl_representative(v, nstat);
        for (int i = nidx[v] + lane; i < nidx[v + 1]; i += ECL_WARPSIZE) {
            const int nli = nlist[i];
            if (v > nli) {
                int ostat = ecl_representative(nli, nstat);
                bool repeat;
                do {
                    repeat = false;
                    if (vstat != ostat) {
                        int ret;
                        if (vstat < ostat) {
                            if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                                ostat = ret;
                                repeat = true;
                            }
                        } else {
                            if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                                vstat = ret;
                                repeat = true;
                            }
                        }
                    }
                } while (repeat);
            }
        }
        if (lane == 0) idx = atomicAdd(&ecl_posL, 1);
        idx = __shfl_sync(0xffffffff, idx, 0);
    }
}

/* ECL-CC: Process high-degree vertices at block granularity */
static __global__ __launch_bounds__(ECL_THREADS_PER_BLOCK, 2048 / ECL_THREADS_PER_BLOCK)
void ecl_compute3(const int nodes, const int32_t* const __restrict__ nidx,
                  const int32_t* const __restrict__ nlist, int32_t* const __restrict__ nstat,
                  const int* const __restrict__ wl)
{
    __shared__ int vB;
    if (threadIdx.x == 0) {
        const int idx = atomicAdd(&ecl_posH, -1);
        vB = (idx > ecl_topH) ? wl[idx] : -1;
    }
    __syncthreads();
    while (vB >= 0) {
        const int v = vB;
        __syncthreads();
        int vstat = ecl_representative(v, nstat);
        for (int i = nidx[v] + threadIdx.x; i < nidx[v + 1]; i += ECL_THREADS_PER_BLOCK) {
            const int nli = nlist[i];
            if (v > nli) {
                int ostat = ecl_representative(nli, nstat);
                bool repeat;
                do {
                    repeat = false;
                    if (vstat != ostat) {
                        int ret;
                        if (vstat < ostat) {
                            if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                                ostat = ret;
                                repeat = true;
                            }
                        } else {
                            if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                                vstat = ret;
                                repeat = true;
                            }
                        }
                    }
                } while (repeat);
            }
        }
        if (threadIdx.x == 0) {
            const int idx = atomicAdd(&ecl_posH, -1);
            vB = (idx > ecl_topH) ? wl[idx] : -1;
        }
        __syncthreads();
    }
}

/* ECL-CC: Link all vertices to sink (final pointer jumping) */
static __global__ __launch_bounds__(ECL_THREADS_PER_BLOCK, 2048 / ECL_THREADS_PER_BLOCK)
void ecl_flatten(const int nodes, int32_t* const __restrict__ nstat)
{
    const int from = threadIdx.x + blockIdx.x * ECL_THREADS_PER_BLOCK;
    const int incr = gridDim.x * ECL_THREADS_PER_BLOCK;

    for (int v = from; v < nodes; v += incr) {
        int next, vstat = nstat[v];
        const int old = vstat;
        while (vstat > (next = nstat[vstat])) {
            vstat = next;
        }
        if (old != vstat) nstat[v] = vstat;
    }
}

/* ========================================================================
 * Afforest Algorithm Kernels
 * Based on: Sutton, Ben-Nun, Hoefler, "Optimizing Parallel Graph Connectivity
 * Computation via Subgraph Sampling", IPDPS 2018
 * ======================================================================== */

#define AFFOREST_SAMPLE_SIZE 2  /* Number of neighbors to sample per vertex */

/* Afforest: Find representative with path compression */
static inline __device__ int afforest_find(int v, int32_t* const __restrict__ labels)
{
    int curr = v;
    while (curr != labels[curr]) {
        int next = labels[curr];
        /* Path compression: shortcut to grandparent */
        labels[curr] = labels[next];
        curr = next;
    }
    return curr;
}

/* Afforest: Union operation - link two trees */
static inline __device__ void afforest_unite(int u, int v, int32_t* const __restrict__ labels)
{
    int root_u = afforest_find(u, labels);
    int root_v = afforest_find(v, labels);

    while (root_u != root_v) {
        /* Always link higher-numbered root to lower */
        int high = root_u > root_v ? root_u : root_v;
        int low = root_u < root_v ? root_u : root_v;

        /* Try to update the parent of high to low */
        if (atomicCAS(&labels[high], high, low) == high) {
            break;  /* Success */
        }
        /* CAS failed, re-find roots */
        root_u = afforest_find(u, labels);
        root_v = afforest_find(v, labels);
    }
}

/* Afforest Phase 1: Sampling - process first k neighbors for all vertices */
static __global__ void afforest_sample_kernel(
    const int nodes,
    const int32_t* const __restrict__ row_ptr,
    const int32_t* const __restrict__ col_idx,
    int32_t* const __restrict__ labels,
    const int sample_size)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        const int beg = __ldg(&row_ptr[v]);
        const int end = __ldg(&row_ptr[v + 1]);
        const int deg = end - beg;

        /* Process at most sample_size neighbors */
        const int limit = min(deg, sample_size);
        for (int i = 0; i < limit; i++) {
            const int neighbor = __ldg(&col_idx[beg + i]);
            afforest_unite(v, neighbor, labels);
        }
    }
}

/* Afforest: Compress all paths to roots */
static __global__ void afforest_compress_kernel(
    const int nodes,
    int32_t* const __restrict__ labels)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        /* Find root and compress path */
        int root = v;
        while (root != labels[root]) {
            root = labels[root];
        }
        /* Point directly to root */
        if (labels[v] != root) {
            labels[v] = root;
        }
    }
}

/* Afforest Phase 2: Link remaining edges, skipping vertices already in large component */
static __global__ void afforest_link_kernel(
    const int nodes,
    const int32_t* const __restrict__ row_ptr,
    const int32_t* const __restrict__ col_idx,
    int32_t* const __restrict__ labels,
    const int sample_size,
    const int32_t skip_component)  /* Skip vertices belonging to largest component */
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        /* Skip if this vertex is already in the large component (optimization) */
        if (labels[v] == skip_component) {
            continue;
        }

        const int beg = __ldg(&row_ptr[v]);
        const int end = __ldg(&row_ptr[v + 1]);

        /* Process remaining edges (skip first sample_size) */
        for (int i = beg + sample_size; i < end; i++) {
            const int neighbor = __ldg(&col_idx[i]);
            /* Unite with ALL neighbors, including those in large component */
            afforest_unite(v, neighbor, labels);
        }
    }
}

/* Afforest: Link ALL remaining edges (when no large component found) */
static __global__ void afforest_link_all_kernel(
    const int nodes,
    const int32_t* const __restrict__ row_ptr,
    const int32_t* const __restrict__ col_idx,
    int32_t* const __restrict__ labels,
    const int sample_size)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        const int beg = __ldg(&row_ptr[v]);
        const int end = __ldg(&row_ptr[v + 1]);

        /* Process remaining edges (skip first sample_size already processed) */
        for (int i = beg + sample_size; i < end; i++) {
            const int neighbor = __ldg(&col_idx[i]);
            afforest_unite(v, neighbor, labels);
        }
    }
}

/* Kernel to find the most frequent component (largest component) using atomics */
static __global__ void afforest_find_largest_kernel(
    const int nodes,
    const int32_t* const __restrict__ labels,
    int32_t* const __restrict__ component_counts,
    const int max_components)
{
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    for (int v = from; v < nodes; v += incr) {
        const int32_t comp = labels[v];
        if (comp < max_components) {
            atomicAdd(&component_counts[comp], 1);
        }
    }
}

/* Kernel to find max in component_counts array */
static __global__ void afforest_reduce_max_kernel(
    const int32_t* const __restrict__ counts,
    const int n,
    int32_t* __restrict__ max_comp,
    int32_t* __restrict__ max_count)
{
    __shared__ int32_t s_max_comp[256];
    __shared__ int32_t s_max_count[256];

    const int tid = threadIdx.x;
    const int from = threadIdx.x + blockIdx.x * blockDim.x;
    const int incr = gridDim.x * blockDim.x;

    int32_t local_max_count = 0;
    int32_t local_max_comp = -1;

    for (int i = from; i < n; i += incr) {
        int32_t c = counts[i];
        if (c > local_max_count) {
            local_max_count = c;
            local_max_comp = i;
        }
    }

    s_max_comp[tid] = local_max_comp;
    s_max_count[tid] = local_max_count;
    __syncthreads();

    /* Reduction in shared memory */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max_count[tid + s] > s_max_count[tid]) {
                s_max_count[tid] = s_max_count[tid + s];
                s_max_comp[tid] = s_max_comp[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_count, s_max_count[0]);
        /* Store the component ID if this block found the max */
        if (s_max_count[0] == *max_count) {
            *max_comp = s_max_comp[0];
        }
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

    double t_start = get_time();

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
    double t_mem_start = get_time();
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
    double t_mem_end = get_time();
    printf("GPU memory allocation and transfer: %.5f seconds\n", t_mem_end - t_mem_start);

    /* Configure kernel launch parameters */
    int32_t num_blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Initialize labels on device */
    init_labels_kernel<<<num_blocks, BLOCK_SIZE>>>(d_labels, num_vertices);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Label propagation iterations */
    double t_prop_start = get_time();
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
    double t_prop_end = get_time();

    printf("Label propagation converged in %d iterations (%.5f seconds)\n",
           result->num_iterations, t_prop_end - t_prop_start);

    /* Compress labels (pointer jumping to find roots) */
    double t_compress_start = get_time();
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
    double t_compress_end = get_time();

    printf("Label compression completed in %d iterations (%.5f seconds)\n",
           compress_iterations, t_compress_end - t_compress_start);

    /* Copy labels back to host */
    double t_copy_start = get_time();
    CUDA_CHECK(cudaMemcpy(result->labels, d_labels, sizeof(int32_t) * num_vertices,
                          cudaMemcpyDeviceToHost));
    double t_copy_end = get_time();
    printf("Copy results to host: %.5f seconds\n", t_copy_end - t_copy_start);

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

void cc_cuda_result_destroy(CCResult *result) {
    if (result == NULL) {
        return;
    }
    free(result->labels);
    free(result);
}

CCResult *cc_cuda_ecl(const Graph *restrict g) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);
    const int32_t num_edges = graph_get_num_edges(g);
    /* Total edges in CSR (undirected = 2 * edges) */
    const int32_t total_edges = 2 * num_edges;

    if (num_vertices <= 0) {
        fprintf(stderr, "Error: Invalid number of vertices\n");
        return NULL;
    }

    printf("Running ECL-CC CUDA on %d vertices, %d edges\n", num_vertices, num_edges);

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

    /* Get device properties */
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int SMs = deviceProp.multiProcessorCount;
    const int mTSM = deviceProp.maxThreadsPerMultiProcessor;
    printf("ECL-CC using GPU: %s with %d SMs\n", deviceProp.name, SMs);

    /* Allocate device memory */
    int32_t *d_row_ptr = NULL;
    int32_t *d_col_idx = NULL;
    int32_t *d_labels = NULL;
    int *d_worklist = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_row_ptr, sizeof(int32_t) * (num_vertices + 1)));
    CUDA_CHECK(cudaMalloc((void **)&d_col_idx, sizeof(int32_t) * total_edges));
    CUDA_CHECK(cudaMalloc((void **)&d_labels, sizeof(int32_t) * num_vertices));
    CUDA_CHECK(cudaMalloc((void **)&d_worklist, sizeof(int) * num_vertices));

    /* Copy graph data to device */
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g->row_ptr, sizeof(int32_t) * (num_vertices + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g->col_idx, sizeof(int32_t) * total_edges, cudaMemcpyHostToDevice));

    /* Set cache preference for L1 */
    cudaFuncSetCacheConfig(ecl_init, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(ecl_compute1, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(ecl_compute2, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(ecl_compute3, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(ecl_flatten, cudaFuncCachePreferL1);

    /* Calculate optimal number of blocks */
    const int blocks = SMs * mTSM / ECL_THREADS_PER_BLOCK;

    /* Run ECL-CC algorithm */

    /* Phase 1: Initialize labels with first smaller neighbor */
    ecl_init<<<blocks, ECL_THREADS_PER_BLOCK>>>(num_vertices, d_row_ptr, d_col_idx, d_labels);
    CUDA_CHECK(cudaGetLastError());

    /* Phase 2: Process low-degree vertices (thread granularity) and build worklists */
    ecl_compute1<<<blocks, ECL_THREADS_PER_BLOCK>>>(num_vertices, d_row_ptr, d_col_idx, d_labels, d_worklist);
    CUDA_CHECK(cudaGetLastError());

    /* Phase 3: Process medium-degree vertices (warp granularity) */
    ecl_compute2<<<blocks, ECL_THREADS_PER_BLOCK>>>(num_vertices, d_row_ptr, d_col_idx, d_labels, d_worklist);
    CUDA_CHECK(cudaGetLastError());

    /* Phase 4: Process high-degree vertices (block granularity) */
    ecl_compute3<<<blocks, ECL_THREADS_PER_BLOCK>>>(num_vertices, d_row_ptr, d_col_idx, d_labels, d_worklist);
    CUDA_CHECK(cudaGetLastError());

    /* Phase 5: Final pointer jumping to flatten all paths */
    ecl_flatten<<<blocks, ECL_THREADS_PER_BLOCK>>>(num_vertices, d_labels);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy results back to host */
    CUDA_CHECK(cudaMemcpy(result->labels, d_labels, sizeof(int32_t) * num_vertices, cudaMemcpyDeviceToHost));

    /* Count components */
    result->num_components = count_components(result->labels, num_vertices);
    result->num_iterations = 1; /* ECL-CC is single-pass */
    printf("Found %d connected components\n", result->num_components);

    /* Cleanup device memory */
    cudaFree(d_worklist);
    cudaFree(d_labels);
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);

    return result;
}

CCResult *cc_cuda_afforest(const Graph *restrict g) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);
    const int32_t num_edges = graph_get_num_edges(g);
    const int32_t total_edges = 2 * num_edges;

    if (num_vertices <= 0) {
        fprintf(stderr, "Error: Invalid number of vertices\n");
        return NULL;
    }

    printf("Running Afforest CUDA on %d vertices, %d edges\n", num_vertices, num_edges);

    /* Allocate result structure */
    CCResult *result = (CCResult *)malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        return NULL;
    }

    result->labels = (int32_t *)malloc(sizeof(int32_t) * num_vertices);
    if (result->labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate labels array\n");
        free(result);
        return NULL;
    }

    /* Get device properties */
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    const int SMs = deviceProp.multiProcessorCount;
    const int mTSM = deviceProp.maxThreadsPerMultiProcessor;
    printf("Afforest using GPU: %s with %d SMs\n", deviceProp.name, SMs);

    /* Allocate device memory */
    int32_t *d_row_ptr = NULL;
    int32_t *d_col_idx = NULL;
    int32_t *d_labels = NULL;
    int32_t *d_comp_counts = NULL;
    int32_t *d_max_comp = NULL;
    int32_t *d_max_count = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_row_ptr, sizeof(int32_t) * (num_vertices + 1)));
    CUDA_CHECK(cudaMalloc((void **)&d_col_idx, sizeof(int32_t) * total_edges));
    CUDA_CHECK(cudaMalloc((void **)&d_labels, sizeof(int32_t) * num_vertices));
    CUDA_CHECK(cudaMalloc((void **)&d_comp_counts, sizeof(int32_t) * num_vertices));
    CUDA_CHECK(cudaMalloc((void **)&d_max_comp, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc((void **)&d_max_count, sizeof(int32_t)));

    /* Copy graph data to device */
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g->row_ptr, sizeof(int32_t) * (num_vertices + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g->col_idx, sizeof(int32_t) * total_edges, cudaMemcpyHostToDevice));

    /* Initialize labels: label[v] = v */
    const int blocks = SMs * mTSM / BLOCK_SIZE;
    init_labels_kernel<<<blocks, BLOCK_SIZE>>>(d_labels, num_vertices);
    CUDA_CHECK(cudaGetLastError());

    /* Phase 1: Sampling - process first AFFOREST_SAMPLE_SIZE neighbors */
    afforest_sample_kernel<<<blocks, BLOCK_SIZE>>>(
        num_vertices, d_row_ptr, d_col_idx, d_labels, AFFOREST_SAMPLE_SIZE);
    CUDA_CHECK(cudaGetLastError());

    /* Compress paths after sampling */
    afforest_compress_kernel<<<blocks, BLOCK_SIZE>>>(num_vertices, d_labels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Find largest component to potentially skip in Phase 2 */
    CUDA_CHECK(cudaMemset(d_comp_counts, 0, sizeof(int32_t) * num_vertices));
    int32_t h_max_comp = -1;
    int32_t h_max_count = 0;
    CUDA_CHECK(cudaMemcpy(d_max_comp, &h_max_comp, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_max_count, &h_max_count, sizeof(int32_t), cudaMemcpyHostToDevice));

    /* Count vertices in each component */
    afforest_find_largest_kernel<<<blocks, BLOCK_SIZE>>>(
        num_vertices, d_labels, d_comp_counts, num_vertices);
    CUDA_CHECK(cudaGetLastError());

    /* Find the largest component */
    afforest_reduce_max_kernel<<<blocks, 256>>>(
        d_comp_counts, num_vertices, d_max_comp, d_max_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_max_comp, d_max_comp, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_max_count, d_max_count, sizeof(int32_t), cudaMemcpyDeviceToHost));

    /* Phase 2: Process remaining edges */
    /* Threshold: skip large component optimization if it contains > 50% of vertices */
    const float large_threshold = 0.5f;
    const bool has_large_component = (h_max_count > (int32_t)(num_vertices * large_threshold));

    if (has_large_component && h_max_comp >= 0) {
        /* Skip vertices in the largest component */
        afforest_link_kernel<<<blocks, BLOCK_SIZE>>>(
            num_vertices, d_row_ptr, d_col_idx, d_labels, AFFOREST_SAMPLE_SIZE, h_max_comp);
    } else {
        /* No dominant large component, process all remaining edges */
        afforest_link_all_kernel<<<blocks, BLOCK_SIZE>>>(
            num_vertices, d_row_ptr, d_col_idx, d_labels, AFFOREST_SAMPLE_SIZE);
    }
    CUDA_CHECK(cudaGetLastError());

    /* Final path compression */
    afforest_compress_kernel<<<blocks, BLOCK_SIZE>>>(num_vertices, d_labels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy results back to host */
    CUDA_CHECK(cudaMemcpy(result->labels, d_labels, sizeof(int32_t) * num_vertices, cudaMemcpyDeviceToHost));

    /* Count components */
    result->num_components = count_components(result->labels, num_vertices);
    result->num_iterations = 2; /* Afforest has 2 phases */
    printf("Found %d connected components\n", result->num_components);

    /* Cleanup */
    cudaFree(d_max_count);
    cudaFree(d_max_comp);
    cudaFree(d_comp_counts);
    cudaFree(d_labels);
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);

    return result;
}







