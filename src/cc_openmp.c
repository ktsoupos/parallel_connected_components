#include "cc_openmp.h"
#include "cc_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

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
    int32_t *labels = malloc(sizeof(int32_t) * (size_t) num_vertices);

    if (labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate labels arrays\n");
        free(labels);
        free(result);
        return NULL;
    }

    /* Initialize: each vertex gets its own label */
#pragma omp parallel for default(none) shared(labels, num_vertices)
    for (int32_t i = 0; i < num_vertices; i++) {
        labels[i] = i;
    }

    bool changed = false;
    result->num_iterations = 0;
    const int32_t max_iterations = num_vertices;

    /* Note: Static analyzers may warn about this loop, but the OpenMP reduction
     * clause correctly updates 'changed' across threads, allowing proper termination. max_iterations.
     * max_iterations signals compiler that the loop will be terminated */
    do {
        result->num_iterations++;
        changed = false;

        /* All vertices update their labels in parallel */
#pragma omp parallel for default(none) shared(g, labels, num_vertices) reduction(||:changed)
        for (int32_t v = 0; v < num_vertices; v++) {
            int32_t num_neighbors;
            const int32_t *restrict neighbors = graph_get_neighbors(g, v, &num_neighbors);

            if (neighbors == NULL) {
                continue;
            }

            /* Find minimum label among neighbors */
            int32_t min_label = labels[v];
            for (int32_t j = 0; j < num_neighbors; j++) {
                const int32_t u = neighbors[j];
                if (labels[u] < min_label) {
                    min_label = labels[u];
                }
            }

            /* Track if any change occurred */
            if (min_label < labels[v]) {
#pragma omp atomic write
                labels[v] = min_label;
                changed = true;
            }
            /* Propagate label to the neighbors*/
            for (int32_t j = 0; j < num_neighbors; j++) {
                const int32_t u = neighbors[j];
                if (labels[u] > min_label) {
#pragma omp atomic write
                    labels[u] = min_label;
                    changed = true;
                }
            }
        }
    } while (changed && result->num_iterations < max_iterations);

    /* Store final labels in result */
    result->labels = labels;

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
 * Hook phase helper: Attempts to hook smaller component roots to neighbors
 * Returns true if any hooking occurred
 * (Ignore static analyser warning)
 */
static bool hook_phase(const Graph *restrict g, int32_t *restrict parents, const int32_t num_vertices) {
    bool hooking = false;

    /* For all vertices in parallel */
#pragma omp parallel for default(none) shared(g, parents, num_vertices) reduction(||:hooking)
    for (int32_t u = 0; u < num_vertices; u++) {
        int32_t num_neighbors = 0;
        const int32_t *restrict neighbors = graph_get_neighbors(g, u, &num_neighbors);

        if (neighbors == NULL) {
            continue;
        }

        /* For all neighbors of u */
        for (int32_t j = 0; j < num_neighbors; j++) {
            const int32_t v = neighbors[j];

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
    /* For all vertices in parallel */
#pragma omp parallel for default(none) shared(parents, num_vertices)
    for (int32_t v = 0; v < num_vertices; v++) {
        /* Path compression: follow parent pointers until reaching root */
        while (parents[parents[v]] != parents[v]) {
            parents[v] = parents[parents[v]];
        }
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

    /* Allocate parent array (π in the algorithm) */
    int32_t *parents = malloc(sizeof(int32_t) * (size_t) num_vertices);

    if (parents == NULL) {
        fprintf(stderr, "Error: Failed to allocate parent array\n");
        free(result);
        return NULL;
    }

    /* Initialize: each vertex is its own parent (π(v) ← v) */
#pragma omp parallel for default(none) shared(parents, num_vertices)
    for (int32_t i = 0; i < num_vertices; i++) {
        parents[i] = i;
    }

    /* Main Shiloach-Vishkin loop */
    result->num_iterations = 0;
    bool hooking = true;
    const int32_t max_iterations = num_vertices; // Safety bound

    while (hooking && result->num_iterations < max_iterations) { // ignore compiler warning  on hooking
        result->num_iterations++;

        /* Hook phase: try to connect components */
        hooking = hook_phase(g, parents, num_vertices);

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
