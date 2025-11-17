#include "cc_pthreads.h"

#include "cc_common.h"
#include "definitions.h"
#include "deque.h"
#include "threadpool.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

/* Simplified label propagation using basic pthreads (no work-stealing) */
typedef struct {
    int thread_id;
    int num_threads;
    const Graph *g;
    _Atomic(int32_t) *labels;
    int32_t num_vertices;
    _Atomic(int8_t) *changed_flag;
} lp_thread_args_t __attribute__((aligned(64)));

/**
 * Thread function for one iteration of label propagation
 */
static void *lp_iteration_thread(void *arg) {
    lp_thread_args_t *args = (lp_thread_args_t *)arg;
    bool local_changed = false;

    /* Static scheduling */
    const int32_t verts_per_thread = args->num_vertices / args->num_threads;
    const int32_t start = args->thread_id * verts_per_thread;
    int32_t end = start + verts_per_thread;

    /* Last thread takes remaining vertices */
    if (args->thread_id == args->num_threads - 1) {
        end = args->num_vertices;
    }

    /* Process assigned vertices */
    for (int32_t v = start; v < end; v++) {
        int32_t current_label = atomic_load_explicit(&args->labels[v], memory_order_relaxed);
        int32_t min_label = current_label;

        /* Check all neighbors */
        int32_t num_neighbors = 0;
        const int32_t *neighbors = graph_get_neighbors(args->g, v, &num_neighbors);

        for (int32_t i = 0; i < num_neighbors; i++) {
            int32_t neighbor_label = atomic_load_explicit(&args->labels[neighbors[i]], memory_order_acquire);
            if (neighbor_label < min_label) {
                min_label = neighbor_label;
            }
        }

        /* Update if we found a smaller label */
        if (min_label < current_label) {
            atomic_store_explicit(&args->labels[v], min_label, memory_order_release);
            local_changed = true;
        }
    }

    /* Report if any changes occurred */
    if (local_changed) {
        atomic_store_explicit(args->changed_flag, 1, memory_order_release);
    }

    return NULL;
}

/**
 * Simple synchronized label propagation (no work-stealing - much simpler and more reliable)
 */
CCResult *label_propagation_sync_pthreads(const Graph *g, int32_t num_threads) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    if (num_threads <= 0) {
        fprintf(stderr, "Error: Invalid number of threads\n");
        return NULL;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);
    if (num_vertices <= 0) {
        fprintf(stderr, "Error: Invalid number of vertices\n");
        return NULL;
    }

    /* Allocate atomic labels array */
    _Atomic(int32_t) *labels = malloc(sizeof(_Atomic(int32_t)) * (size_t)num_vertices);
    if (labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate labels array\n");
        return NULL;
    }

    /* Initialize labels (each vertex starts with its own ID) */
    for (int32_t i = 0; i < num_vertices; i++) {
        atomic_init(&labels[i], i);
    }

    /* Allocate thread resources */
    pthread_t *threads = malloc(sizeof(pthread_t) * (size_t)num_threads);
    lp_thread_args_t *args = malloc(sizeof(lp_thread_args_t) * (size_t)num_threads);
    _Atomic(int8_t) changed_flag;
    atomic_init(&changed_flag, 0);

    if (threads == NULL || args == NULL) {
        fprintf(stderr, "Error: Failed to allocate thread resources\n");
        free(labels);
        free(threads);
        free(args);
        return NULL;
    }

    /* Initialize thread arguments */
    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].num_threads = num_threads;
        args[i].g = g;
        args[i].labels = labels;
        args[i].num_vertices = num_vertices;
        args[i].changed_flag = &changed_flag;
    }

    /* Iterative label propagation */
    int32_t num_iterations = 0;
    const int32_t MAX_ITERATIONS = 1000;

    while (num_iterations < MAX_ITERATIONS) {
        atomic_store_explicit(&changed_flag, 0, memory_order_relaxed);

        /* Launch threads for this iteration */
        for (int i = 0; i < num_threads; i++) {
            pthread_create(&threads[i], NULL, lp_iteration_thread, &args[i]);
        }

        /* Wait for all threads to complete */
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        num_iterations++;

        /* Check for convergence */
        if (atomic_load_explicit(&changed_flag, memory_order_acquire) == 0) {
            break; /* Converged! */
        }
    }

    /* Copy atomic labels to regular array */
    int32_t *final_labels = malloc(sizeof(int32_t) * (size_t)num_vertices);
    if (final_labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate final labels\n");
        free(labels);
        free(threads);
        free(args);
        return NULL;
    }

    for (int32_t i = 0; i < num_vertices; i++) {
        final_labels[i] = atomic_load_explicit(&labels[i], memory_order_relaxed);
    }

    /* Cleanup */
    free(labels);
    free(threads);
    free(args);

    /* Count components */
    const int32_t num_components = count_unique_labels(final_labels, num_vertices);

    /* Create result */
    CCResult *result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate result\n");
        free(final_labels);
        return NULL;
    }

    result->labels = final_labels;
    result->num_components = num_components;
    result->num_iterations = num_iterations;

    return result;
}