#include "cc_pthreads.h"

#include "cc_common.h"
#include "definitions.h"
#include "deque.h"
#include "threadpool.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct SyncContext {
    const Graph *graph;
    _Atomic(int32_t) *labels;
    _Atomic(int8_t) changed_flag;
    Task *task_pool;
    int32_t num_tasks;
    int32_t chunk_size;
} SyncContext;


static void process_chunk_sync(Task *task) {
    if (task == NULL) {
        return;
    }
    SyncContext *context = (SyncContext *)task->context;
    bool local_changed = false;

    for (int32_t v = task->start_vertex; v < task->end_vertex; v++) {
        int32_t current_label = atomic_load_explicit(&context->labels[v], memory_order_relaxed);
        int32_t min_label = current_label;

        int32_t num_neighbors = 0;
        const int32_t *restrict neighbors = graph_get_neighbors(context->graph, v, &num_neighbors);
        for (int32_t neighbor = 0; neighbor < num_neighbors; neighbor++) {
            int32_t neighbor_label = atomic_load_explicit(&context->labels[neighbors[neighbor]],
                                                          memory_order_acquire);

            if (neighbor_label < min_label) {
                min_label = neighbor_label;
            }
        }
        if (min_label < current_label) {
            atomic_store_explicit(&context->labels[v], min_label, memory_order_release);
            local_changed = true;
        }
    }
    if (local_changed) {
        atomic_store_explicit(&context->changed_flag, 1, memory_order_release);
    }
}


CCResult *label_propagation_sync_pthreads(const Graph *g, int32_t num_threads) {

    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    if (num_threads <= 0) {
        fprintf(stderr, "Error: Invalid number of threads\n");
        return NULL;
    }

    const int32_t num_vertices = g->num_vertices;

    _Atomic(int32_t) *labels = (_Atomic(int32_t) *)malloc(
        (size_t)num_vertices * sizeof(_Atomic(int32_t)));
    if (labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate labels array\n");
        return NULL;
    }

    for (int32_t i = 0; i < num_vertices; i++) {
        atomic_init(&labels[i], i);
    }

    int32_t chunk_size = (num_vertices / (num_threads * 4)) + 1;

    if (chunk_size < 1024) {
        chunk_size = 1024; // Min: 1K vertices (avoid too many small chunks)
    }
    if (chunk_size > 16384) {
        chunk_size = 16384; // Max: 16K vertices (ensure load balancing)
    }

    const int32_t num_chunks = (num_vertices + chunk_size - 1) / chunk_size;

    SyncContext *context = (SyncContext *)malloc(sizeof(SyncContext));
    if (context == NULL) {
        fprintf(stderr, "Error: Failed to allocate sync context\n");
        free(labels);
        return NULL;
    }
    context->graph = g;
    context->labels = labels;
    context->chunk_size = chunk_size;
    context->num_tasks = num_chunks;
    atomic_init(&context->changed_flag, 0);

    context->task_pool = calloc((size_t)num_chunks, sizeof(Task));
    if (context->task_pool == NULL) {
        fprintf(stderr, "Error: Failed to allocate task pool\n");
        free(labels);
        free(context);
        return NULL;
    }

    for (int32_t i = 0; i < num_chunks; i++) {
        Task *task = &context->task_pool[i];

        task->func = process_chunk_sync;
        task->start_vertex = (int32_t)(i * chunk_size);
        task->end_vertex = task->start_vertex + chunk_size;

        // Last chunk: don't go past end of array
        if (task->end_vertex > num_vertices) {
            task->end_vertex = num_vertices;
        }

        task->context = (void *)context;
        task->should_free = false; // Don't free pool tasks
    }

    const int64_t deque_capacity = (num_chunks / num_threads) + 64;
    ThreadPool *pool = threadpool_create(num_threads, deque_capacity);
    if (pool == NULL) {
        fprintf(stderr, "Error: Failed to create threadpool\n");
        free(context->task_pool);
        free(labels);
        free(context);
        return NULL;
    }

    // Start worker threads
    threadpool_start(pool);

    int32_t num_iterations = 0;
    const int32_t MAX_ITERATIONS = 1000; // Safety limit

    while (num_iterations < MAX_ITERATIONS) {
        atomic_store_explicit(&context->changed_flag, 0, memory_order_relaxed);

        for (int32_t i = 0; i < num_chunks; i++) {
            Task *task = &context->task_pool[i];
            const int32_t worker_id = i % num_threads;
            Worker *worker = &pool->workers[worker_id];
            if (!deque_push_bottom(&worker->deque, task)) {
                fprintf(stderr, "Warning: Failed to push task %d\n", i);
                continue; // Skip this task
            }
            atomic_fetch_add_explicit(&pool->active_tasks, 1, memory_order_release);
        }
        threadpool_wake_workers(pool);
        threadpool_barrier(pool);

        num_iterations++;

        const int8_t changed = atomic_load_explicit(&context->changed_flag, memory_order_acquire);

        if (changed == 0) {
            break; // Converged! No changes this iteration
        }
    }
    threadpool_shutdown(pool);
    threadpool_destroy(pool);

    int32_t *final_labels = (int32_t *)malloc((size_t)num_vertices * sizeof(int32_t));
    if (final_labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate final labels\n");
        free(context->task_pool);
        free(labels);
        free(context);
        return NULL;
    }

    // Copy from atomic array to regular array
    for (int32_t i = 0; i < num_vertices; i++) {
        final_labels[i] = atomic_load_explicit(&labels[i], memory_order_relaxed);
    }

    free(context->task_pool); // Free the task pool
    free(labels); // Free atomic labels
    free(context); // Free context

    const int32_t num_components = count_unique_labels(final_labels, num_vertices);

    CCResult *result = (CCResult *)malloc(sizeof(CCResult));
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