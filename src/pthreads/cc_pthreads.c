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
            int32_t neighbor_label = atomic_load_explicit(&args->labels[neighbors[i]],
                                                          memory_order_acquire);
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

/* ============ ASYNC LABEL PROPAGATION WITH WORK-STEALING ============ */

typedef struct AsyncLPContext {
    const Graph *g;
    _Atomic(int32_t) *labels;
    _Atomic(int8_t) *changed;
    int32_t num_vertices;
} AsyncLPContext;

/**
 * Process a vertex chunk with work-stealing (simplified - no dynamic task creation)
 */
static void process_chunk_async(Task *task) {
    if (task == NULL)
        return;

    AsyncLPContext *ctx = (AsyncLPContext *)task->context;
    bool local_changed = false;

    /* Process range of vertices */
    for (int32_t v = task->start_vertex; v < task->end_vertex; v++) {
        int32_t current_label = atomic_load_explicit(&ctx->labels[v], memory_order_relaxed);
        int32_t min_label = current_label;

        /* Find minimum neighbor label */
        int32_t num_neighbors = 0;
        const int32_t *neighbors = graph_get_neighbors(ctx->g, v, &num_neighbors);

        for (int32_t i = 0; i < num_neighbors; i++) {
            int32_t neighbor_label = atomic_load_explicit(&ctx->labels[neighbors[i]],
                                                          memory_order_acquire);
            if (neighbor_label < min_label) {
                min_label = neighbor_label;
            }
        }

        /* Update if changed */
        if (min_label < current_label) {
            atomic_store_explicit(&ctx->labels[v], min_label, memory_order_release);
            local_changed = true;
        }
    }

    /* Report if any changes */
    if (local_changed) {
        atomic_store_explicit(ctx->changed, 1, memory_order_release);
    }
}

/**
 * Async label propagation with work-stealing (simplified to avoid segfaults)
 * Uses work-stealing for load balancing but with pre-allocated tasks
 */
CCResult *label_propagation_async_pthreads(const Graph *g, int32_t num_threads) {
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

    /* Allocate atomic labels */
    _Atomic(int32_t) *labels = malloc(sizeof(_Atomic(int32_t)) * (size_t)num_vertices);
    if (labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate labels array\n");
        return NULL;
    }

    /* Initialize labels */
    for (int32_t i = 0; i < num_vertices; i++) {
        atomic_init(&labels[i], i);
    }

    /* Setup chunking for work-stealing */
    int32_t chunk_size = num_vertices / (num_threads * 4);
    if (chunk_size < 512)
        chunk_size = 512;
    if (chunk_size > 8192)
        chunk_size = 8192;

    const int32_t num_chunks = (num_vertices + chunk_size - 1) / chunk_size;

    /* Allocate context and task pool */
    AsyncLPContext *ctx = malloc(sizeof(AsyncLPContext));
    Task *task_pool = calloc((size_t)num_chunks, sizeof(Task));
    _Atomic(int8_t) changed;
    atomic_init(&changed, 0);

    if (ctx == NULL || task_pool == NULL) {
        fprintf(stderr, "Error: Failed to allocate context/tasks\n");
        free(labels);
        free(ctx);
        free(task_pool);
        return NULL;
    }

    ctx->g = g;
    ctx->labels = labels;
    ctx->changed = &changed;
    ctx->num_vertices = num_vertices;

    /* Initialize task pool */
    for (int32_t i = 0; i < num_chunks; i++) {
        task_pool[i].func = process_chunk_async;
        task_pool[i].start_vertex = i * chunk_size;
        task_pool[i].end_vertex = (i == num_chunks - 1) ? num_vertices : (i + 1) * chunk_size;
        task_pool[i].context = ctx;
        task_pool[i].should_free = false;
    }

    /* Create threadpool */
    const int64_t deque_capacity = num_chunks * 4;
    ThreadPool *pool = threadpool_create(num_threads, deque_capacity);
    if (pool == NULL) {
        fprintf(stderr, "Error: Failed to create threadpool\n");
        free(labels);
        free(ctx);
        free(task_pool);
        return NULL;
    }

    threadpool_start(pool);

    /* Iterate until convergence */
    int32_t num_iterations = 0;
    const int32_t MAX_ITERATIONS = 100000;

    while (num_iterations < MAX_ITERATIONS) {
        atomic_store_explicit(&changed, 0, memory_order_relaxed);

        /* Distribute tasks to workers for stealing */
        /* Distribute tasks to workers for stealing */
        for (int32_t i = 0; i < num_chunks; i++) {
            const int32_t worker_id = i % num_threads;
            Worker *worker = &pool->workers[worker_id];

            // 1) increase outstanding count BEFORE making task visible:
            atomic_fetch_add_explicit(&pool->active_tasks, 1, memory_order_acq_rel);

            // 2) publish task to deque
            bool pushed = deque_push_bottom(&worker->deque, &task_pool[i]);

            if (!pushed) {
                // fallback: if push failed, undo active_tasks and handle failure
                atomic_fetch_sub_explicit(&pool->active_tasks, 1, memory_order_acq_rel);
                // handle error (e.g., push to another worker, expand deque, abort)
                // For simplicity we print and continue
                fprintf(stderr, "Warning: deque_push_bottom failed for chunk %d\n", i);
                continue;
            }

            // 3) wake workers (while not required to hold work_mutex, doing so
            //    avoids races with waiters doing the check+wait under the same mutex)
            pthread_mutex_lock(&pool->work_mutex);
            pthread_cond_broadcast(&pool->work_available);
            pthread_mutex_unlock(&pool->work_mutex);
        }
        num_iterations++;

        threadpool_wait(pool);

        num_iterations++;

        if (atomic_load_explicit(&changed, memory_order_acquire) == 0) {
            break;   // converged
        }
    }

    /* Shutdown */
    threadpool_shutdown(pool);
    threadpool_destroy(pool);

    /* Copy to regular array */
    int32_t *final_labels = malloc(sizeof(int32_t) * (size_t)num_vertices);
    if (final_labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate final labels\n");
        free(labels);
        free(ctx);
        free(task_pool);
        return NULL;
    }

    for (int32_t i = 0; i < num_vertices; i++) {
        final_labels[i] = atomic_load_explicit(&labels[i], memory_order_relaxed);
    }

    /* Cleanup */
    free(labels);
    free(ctx);
    free(task_pool);

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