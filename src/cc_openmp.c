#include "cc_openmp.h"
#include "cc_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>


int set_omp_threads(int num_threads) {
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);
    return num_threads;
}

int get_omp_threads(void) {
    return omp_get_max_threads();
}

void openmp_hello_world(void) {
    printf("\n=== OpenMP Hello World ===\n");
    printf("Max threads available: %d\n", omp_get_max_threads());
    printf("Running parallel region with %d threads:\n\n", omp_get_max_threads());

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const int total_threads = omp_get_num_threads();

#pragma omp critical
        {
            printf("  Hello from thread %d of %d\n", thread_id, total_threads);
        }
    }

    printf("\nParallel region completed!\n");
}

CCResult *label_propagation_sync_omp(const Graph *restrict g, const int num_threads) {
    /* Check arguments */
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);
    if (num_vertices <= 0) {
        fprintf(stderr, "Error: Invalid number of vertices\n");
        return NULL;
    }

    /* Set number of threads */
    set_omp_threads(num_threads);

    /* Allocate result structure */
    CCResult *restrict result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        return NULL;
    }

    /* Allocate labels arrays - need two for double buffering */
    int32_t *labels_curr = malloc(sizeof(int32_t) * (size_t) num_vertices);
    int32_t *labels_next = malloc(sizeof(int32_t) * (size_t) num_vertices);

    if (labels_curr == NULL || labels_next == NULL) {
        fprintf(stderr, "Error: Failed to allocate labels arrays\n");
        free(labels_curr);
        free(labels_next);
        free(result);
        return NULL;
    }

    /* Initialize: each vertex gets its own label */
#pragma omp parallel for default(none) shared(labels_curr, num_vertices)
    for (int32_t i = 0; i < num_vertices; i++) {
        labels_curr[i] = i;
    }

    /* Label propagation with synchronous updates */
    result->num_iterations = 0;
    bool changed = false;
    const int32_t max_iterations = num_vertices; // Convergence bound: at most n iterations

    /* Note: Static analyzers may warn about this loop, but the OpenMP reduction
     * clause correctly updates 'changed' across threads, allowing proper termination. max_iterations.
     * max_iterations signals compiler that the loop will be terminated*/
    do {
        result->num_iterations++;
        changed = false;

        /* All vertices update their labels in parallel */
#pragma omp parallel for default(none) shared(g, labels_curr, labels_next, num_vertices) reduction(||:changed)
        for (int32_t v = 0; v < num_vertices; v++) {
            int32_t num_neighbors;
            const int32_t *restrict neighbors = graph_get_neighbors(g, v, &num_neighbors);

            if (neighbors == NULL) {
                labels_next[v] = labels_curr[v];
                continue;
            }

            /* Find minimum label among neighbors */
            int32_t min_label = labels_curr[v];
            for (int32_t j = 0; j < num_neighbors; j++) {
                const int32_t u = neighbors[j];
                if (labels_curr[u] < min_label) {
                    min_label = labels_curr[u];
                }
            }

            /* Update next buffer */
            labels_next[v] = min_label;

            /* Track if any change occurred */
            if (min_label < labels_curr[v]) {
                changed = true;
            }
        }

        /* Swap buffers */
        int32_t *temp = labels_curr;
        labels_curr = labels_next;
        labels_next = temp;
    } while (changed && result->num_iterations < max_iterations);

    /* Store final labels in result */
    result->labels = labels_curr;
    free(labels_next);

    /* Count connected components */
    result->num_components = count_unique_labels(result->labels, num_vertices);
    if (result->num_components < 0) {
        fprintf(stderr, "Error: Failed to count components\n");
        free(result->labels);
        free(result);
        return NULL;
    }

    return result;
}

CCResult *label_propagation_async_omp(const Graph *restrict g, int num_threads) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);
    if (num_vertices <= 0) {
        fprintf(stderr, "Error: Invalid number of vertices\n");
        return NULL;
    }

    /* Configure OpenMP threads */
    set_omp_threads(num_threads);

    /* Allocate result */
    CCResult *restrict result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        return NULL;
    }

    /* Allocate aligned label array for better cache performance */
    int32_t *restrict labels = aligned_alloc(64, sizeof(int32_t) * (size_t) num_vertices);
    if (labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate label array\n");
        free(result);
        return NULL;
    }

    /* Initialize: each vertex is its own label */
#pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < num_vertices; i++) {
        labels[i] = i;
    }

    result->num_iterations = 0;
    const int32_t max_iterations = num_vertices;
    bool changed = true;

    /* --- Label Propagation with Asynchronous Updates --- */
    while (changed && result->num_iterations < max_iterations) {
        result->num_iterations++;
        changed = false;

#pragma omp parallel
        {
            bool local_changed = false;

#pragma omp for schedule(dynamic, 128) nowait
            for (int32_t v = 0; v < num_vertices; v++) {
                const int start = g->row_ptr[v];
                const int end = g->row_ptr[v + 1];
                const int num_neighbors = end - start;

                if (num_neighbors == 0) {
                    continue;
                }
                int32_t min_label = labels[v];

                // Find minimum label among neighbors
                for (int32_t i = start; i < end; i++) {
                    const int32_t u = g->col_idx[i];
                    if (labels[u] < min_label)
                        min_label = labels[u];
                }

                // Update own label if smaller found
                if (min_label < labels[v]) {
#pragma omp atomic write
                    labels[v] = min_label;
                    local_changed = true;
                }

                // Propagate min_label to neighbors (async)
                for (int32_t i = start; i < end; i++) {
                    const int32_t u = g->col_idx[i];
                    if (labels[u] > min_label) {
#pragma omp atomic write
                        labels[u] = min_label;
                        local_changed = true;
                    }
                }
            }
#pragma omp critical
            {
                changed |= local_changed;
            }
        }
    } // end while

    /* Finalize result */
    result->labels = labels;

    result->num_components = count_unique_labels(labels, num_vertices);
    if (result->num_components < 0) {
        fprintf(stderr, "Error: Failed to count components\n");
        free(labels);
        free(result);
        return NULL;
    }

    return result;
}


/**
 * Hook phase helper: Attempts to hook smaller component roots to neighbors
 * Returns true if any hooking occurred
 * (Ignore static analyser warning)
 */

static bool hook_phase(const int32_t *restrict row_ptr, const int32_t *restrict col_idx, int32_t *restrict parents,
                       const int32_t num_vertices) {
    bool hooking = false;

    /* For all vertices in parallel - use schedule(guided) for load balancing */
    // clang-format off
#pragma omp parallel for default(none) shared(row_ptr, col_idx, parents, num_vertices) reduction(||:hooking) schedule(dynamic, 128)
    // clang-format on
    for (int32_t u = 0; u < num_vertices; u++) {
        /* Direct CSR access for better performance */
        const int32_t start = row_ptr[u];
        const int32_t end = row_ptr[u + 1];

        /* For all neighbors of u */
        for (int32_t j = start; j < end; j++) {
            const int32_t v = col_idx[j];

            /* Read parent values once to avoid race conditions */
            const int32_t pi_u = parents[u];
            const int32_t pi_v = parents[v];
            const int32_t pi_pi_v = parents[pi_v];

            /* Check if π(u) < π(v) and π(v) = π(π(v)) (v is a root) */
            if (pi_u < pi_v && pi_v == pi_pi_v) {
                /* Hook: π(π(v)) ← π(u) */
#pragma omp atomic write
                parents[pi_v] = pi_u;
                hooking = true;
            }
        }
    }

    return hooking;
}

/**
 * Shortcut phase helper: Performs path compression on all vertices
 */
static void shortcut_phase(int32_t *restrict parents, const int32_t num_vertices) {
    /* For all vertices in parallel - use static scheduling for predictable workload */
#pragma omp parallel for default(none) shared(parents, num_vertices) schedule(static)
    for (int32_t v = 0; v < num_vertices; v++) {
        /* Path compression: follow parent pointers until reaching root */
        int32_t current = v;
        int32_t parent = parents[current];

        /* Two-pass approach to reduce thread divergence */
        while (parent != parents[parent]) {
            const int32_t grandparent = parents[parent];
            parents[current] = grandparent;
            current = parent;
            parent = grandparent;
        }

        /* Final update to root */
        parents[v] = parent;
    }
}

CCResult *shiiloach_vishkin(const Graph *restrict g, const int num_threads) {
    /* Check arguments */
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);
    if (num_vertices <= 0) {
        fprintf(stderr, "Error: Invalid number of vertices\n");
        return NULL;
    }

    /* Set number of threads */
    set_omp_threads(num_threads);

    /* Allocate result structure */
    CCResult *restrict result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        return NULL;
    }

    /* Allocate aligned parent array (π in the algorithm) for better cache performance */
    int32_t *parents = aligned_alloc(64, sizeof(int32_t) * (size_t) num_vertices);

    if (parents == NULL) {
        fprintf(stderr, "Error: Failed to allocate parent array\n");
        free(result);
        return NULL;
    }

    /* Initialize: each vertex is its own parent (π(v) ← v) */
#pragma omp parallel for default(none) shared(parents, num_vertices) schedule(static)
    for (int32_t i = 0; i < num_vertices; i++) {
        parents[i] = i;
    }

    /* Main Shiloach-Vishkin loop */
    result->num_iterations = 0;
    bool hooking = true;
    const int32_t max_iterations = num_vertices; // Safety bound

    while (hooking && result->num_iterations < max_iterations) {
        // ignore compiler warning  on hooking
        result->num_iterations++;

        /* Hook phase: try to connect components */
        hooking = hook_phase(g->row_ptr, g->col_idx, parents, num_vertices);

        /* Shortcut phase: path compression */
        shortcut_phase(parents, num_vertices);
    }

    /* Store final labels in result */
    result->labels = parents;

    /* Count connected components */
    result->num_components = count_unique_labels(result->labels, num_vertices);
    if (result->num_components < 0) {
        fprintf(stderr, "Error: Failed to count components\n");
        free(result->labels);
        free(result);
        return NULL;
    }

    return result;
}

/**
 * Link two vertices u and v using union-find with path compression
 * Based on the Link function from the GAP Benchmark Suite Afforest implementation
 */
__attribute__((always_inline)) inline
static void link_vertices(const int32_t u, const int32_t v, int32_t *restrict parents) {
    /* Read parent values */
    int32_t p1 = parents[u];
    int32_t p2 = parents[v];

    while (p1 != p2) {
        const int32_t high = (p1 > p2) ? p1 : p2;
        const int32_t low = (p1 < p2) ? p1 : p2;
        const int32_t p_high = parents[high];
        int32_t expected = high;

        if ((p_high == low) || // Was already 'low'
            (p_high == high && (__atomic_compare_exchange_n( // Succeeded on writing 'low'
                 &parents[high], &expected, low, false,
                 __ATOMIC_SEQ_CST,
                 __ATOMIC_SEQ_CST)))) {
            break;
        }

        p1 = parents[expected]; // Update with actual value after CAS
        p2 = parents[low];
    }
}

static void compress(int32_t *restrict parents, int32_t num_vertices) {
#pragma omp parallel for schedule(static, 2048) // todo check chunk size
    for (int32_t n = 0; n < num_vertices; n++) {
        while (parents[parents[n]] != parents[n]) {
            parents[n] = parents[parents[n]];
        }
    }
}

/**
 * Parallel version of sample_frequent_element
 * Uses OpenMP to parallelize both sampling and max-finding
 */
int32_t sample_frequent_element(const int32_t *comp, const int32_t num_vertices, const int32_t num_samples) {
    if (comp == NULL || num_vertices <= 0 || num_samples <= 0) {
        fprintf(stderr, "Error: Invalid parameters for sample_frequent_element\n");
        return -1;
    }

    /* Allocate counter array for tracking sample counts */
    int32_t *sample_counts = calloc((size_t)num_vertices, sizeof(int32_t));
    if (sample_counts == NULL) {
        fprintf(stderr, "Error: Failed to allocate sample_counts array\n");
        return -1;
    }

    /* Parallel sampling phase */
    const unsigned int base_seed = (unsigned int)(time(NULL) ^ (time_t)(uintptr_t)comp);

#pragma omp parallel
    {
        /* Each thread gets its own seed based on thread ID */
        const int tid = omp_get_thread_num();
        unsigned int thread_seed = base_seed + (unsigned int)tid;

#pragma omp for schedule(static)
        for (int32_t i = 0; i < num_samples; i++) {
            /* Thread-safe random number generation */
            const unsigned int rand_val = rand_r(&thread_seed);
            const int32_t idx = (int32_t)(rand_val % (unsigned int)num_vertices);
            const int32_t component_id = comp[idx];

            /* Bounds check to prevent heap corruption */
            if (component_id >= 0 && component_id < num_vertices) {
                /* Atomic increment to avoid race conditions */
#pragma omp atomic
                sample_counts[component_id]++;
            }
        }
    }

    /* Parallel max-finding phase with manual reduction */
    int32_t most_frequent_id = 0;
    int32_t max_count = 0;

#pragma omp parallel
    {
        /* Each thread finds local maximum */
        int32_t local_max_id = 0;
        int32_t local_max_count = 0;

#pragma omp for schedule(static) nowait
        for (int32_t i = 0; i < num_vertices; i++) {
            if (sample_counts[i] > local_max_count) {
                local_max_count = sample_counts[i];
                local_max_id = i;
            }
        }

        /* Combine thread-local results */
#pragma omp critical
        {
            if (local_max_count > max_count) {
                max_count = local_max_count;
                most_frequent_id = local_max_id;
            }
        }
    }

    /* Calculate and print percentage */
    const float percentage = (float)max_count / (float)num_samples * 100.0f;
    printf("Skipping largest intermediate component (ID: %d, approx. %.1f%% of the graph)\n",
           most_frequent_id, percentage);

    free(sample_counts);
    return most_frequent_id;
}

static int32_t count_unique_labels_openmp(const int32_t* labels, const int32_t num_vertices) {
    if (labels == NULL || num_vertices <= 0) {
        return -1;
    }

    // Sequential bounds check
    for (int32_t v = 0; v < num_vertices; v++) {
        int32_t label = labels[v];
        if (label < 0 || label >= num_vertices) {
            fprintf(stderr, "Error: Invalid label %d at index %d (range 0-%d)\n",
                    label, v, num_vertices - 1);
            return -1;
        }
    }

    bool* seen = calloc((size_t)num_vertices, sizeof(bool));
    if (seen == NULL) {
        fprintf(stderr, "Error: Failed to allocate seen array\n");
        return -1;
    }

    int32_t count = 0;

#pragma omp parallel for default(none) shared(labels, num_vertices, seen, count) schedule(static)
    for (int32_t v = 0; v < num_vertices; v++) {
        int32_t label = labels[v];

        // benign race: multiple threads may write 'true', which is safe
        if (!seen[label]) {
            seen[label] = true;

            // safe increment using atomic
#pragma omp atomic
            count++;
        }
    }

    free(seen);
    return count;
}

CCResult *afforest(const Graph *restrict g, int num_threads, int32_t neighbor_rounds) {
    /* Check arguments */
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);
    if (num_vertices <= 0) {
        fprintf(stderr, "Error: Invalid number of vertices\n");
        return NULL;
    }

    /* Set number of threads */
    set_omp_threads(num_threads);

    /* Allocate result structure */
    CCResult *restrict result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        return NULL;
    }

    /* Allocate aligned parent array (π in the algorithm) for better cache performance */
    int32_t *parents = aligned_alloc(64, sizeof(int32_t) * (size_t) num_vertices);

    if (parents == NULL) {
        fprintf(stderr, "Error: Failed to allocate parent array\n");
        free(result);
        return NULL;
    }

    /* Initialize: each vertex is its own parent (π(v) ← v) */
#pragma omp parallel for default(none) shared(parents, num_vertices) schedule(static)
    for (int32_t i = 0; i < num_vertices; i++) {
        parents[i] = i;
    }

    /* Set default neighbor_rounds if needed */
    if (neighbor_rounds <= 0) {
        neighbor_rounds = 2; // Default: sample first 2 neighbors
    }

    for (int32_t r = 0; r < neighbor_rounds; ++r) {
#pragma omp parallel for
        for (int32_t u = 0; u < num_vertices; u++) {
            const int32_t start = g->row_ptr[u];
            const int32_t end = g->row_ptr[u + 1];
            const int32_t num_neighbors = end - start;
            if (r < num_neighbors) {
                const int32_t v = g->col_idx[start + r]; // get the r-th neighbor
                link_vertices(u, v, parents);
            }
        }
        compress(parents, num_vertices);
    }

    const int32_t largest_component = sample_frequent_element(parents, num_vertices, 1024);
//clang-format off
#pragma omp parallel for schedule(dynamic, 2048)
    //clang-format on
    for (int32_t u = 0; u < num_vertices; u++) {
        if (parents[u] == largest_component) continue;

        const int32_t start = g->row_ptr[u];
        const int32_t end = g->row_ptr[u + 1];

        /* Process remaining neighbors (after neighbor_rounds) */
        for (int32_t j = start + neighbor_rounds; j < end; j++) {
            const int32_t v = g->col_idx[j];
            link_vertices(u, v, parents);
        }
    }
    compress(parents, num_vertices);

    /* Store final labels in result */
    result->labels = parents;
    result->num_iterations = neighbor_rounds + 1; // Sampling rounds + final phase

    /* Count connected components */
    result->num_components = count_unique_labels_openmp(result->labels, num_vertices);
    if (result->num_components < 0) {
        fprintf(stderr, "Error: Failed to count components\n");
        free(result->labels);
        free(result);
        return NULL;
    }

    return result;
}
