#include "cc_pthreads.h"

#include "cc_common.h"
#include "definitions.h"
#include "deque.h"
#include "threadpool.h"
#include "worker.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

/* Simplified label propagation using basic pthreads (no work-stealing) */
typedef struct {
    _Atomic(int8_t) *changed_flag;
    int thread_id;
    int num_threads;
    _Atomic(int32_t) *labels;
    int32_t num_vertices;
    const Graph *g;
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

/* Subdivision threshold: if a chunk has more vertices than this, subdivide it */
#define SUBDIVISION_THRESHOLD 8192  /* Increased to reduce subdivision on large graphs */

/* Maximum subdivisions per worker */
#define MAX_SUBDIVISIONS_PER_WORKER 2048  /* Reduced to catch issues earlier */

typedef struct AsyncLPContext {
    const Graph *g;
    _Atomic(int32_t) *labels;
    _Atomic(int8_t) *changed;
    int32_t num_vertices;
    ThreadPool *pool;

    /* Per-worker subdivision task pools (eliminates malloc) */
    Task *subdivision_pool;
    _Atomic(int32_t) *worker_alloc_counters;
    int32_t num_workers;

    bool disable_subdivision; /* Skip subdivision for huge graphs */
} AsyncLPContext;

/* Forward declaration */
static void process_chunk_once(Task *task);

/**
 * Process a vertex chunk ONCE with work-stealing and dynamic subdivision
 * Used for subdivided tasks (one-time processing)
 */
static void process_chunk_once(Task *task) {
    if (task == NULL)
        return;

    AsyncLPContext *ctx = (AsyncLPContext *)task->context;
    int32_t range = task->end_vertex - task->start_vertex;

    /* If chunk is large enough, subdivide it (unless disabled for huge graphs) */
    if (!ctx->disable_subdivision && range > SUBDIVISION_THRESHOLD) {
        int32_t mid = task->start_vertex + (range / 2);

        /* Get current worker */
        Worker *self = worker_current();
        if (self == NULL) {
            goto process_sequentially;
        }

        /* Allocate from per-worker pool (NO MALLOC!) */
        const int32_t worker_id = self->id;
        int32_t alloc_idx = atomic_fetch_add_explicit(
            &ctx->worker_alloc_counters[worker_id], 1, memory_order_relaxed);

        if (alloc_idx >= MAX_SUBDIVISIONS_PER_WORKER) {
            /* Pool exhausted - fallback to sequential */
            atomic_fetch_sub_explicit(&ctx->worker_alloc_counters[worker_id], 1,
                                      memory_order_relaxed);
            goto process_sequentially;
        }

        /* Get task from pool */
        int32_t pool_idx = worker_id * MAX_SUBDIVISIONS_PER_WORKER + alloc_idx;
        Task *left = &ctx->subdivision_pool[pool_idx];

        /* Initialize left sub-task */
        left->func = process_chunk_once;
        left->start_vertex = task->start_vertex;
        left->end_vertex = mid;
        left->context = ctx;
        left->should_free = false; /* Don't free - it's from the pool! */

        /* Increment active tasks before making task visible */
        atomic_fetch_add_explicit(&ctx->pool->active_tasks, 1, memory_order_acq_rel);

        if (!deque_push_bottom(&self->deque, left)) {
            /* Deque full - rollback and fallback */
            atomic_fetch_sub_explicit(&ctx->pool->active_tasks, 1, memory_order_acq_rel);
            atomic_fetch_sub_explicit(&ctx->worker_alloc_counters[worker_id], 1,
                                      memory_order_relaxed);
            goto process_sequentially;
        }

        /* Process right half directly (tail recursion optimization) */
        task->start_vertex = mid;
        /* Fall through to process right half */
        range = task->end_vertex - task->start_vertex;
    }

process_sequentially:
    /* Process range of vertices sequentially */
    bool local_changed = false;

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

/* Just use process_chunk_once - keep it simple! */

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
    fprintf(stderr, "Initializing %d labels...\n", num_vertices);
    fflush(stderr);
    for (int32_t i = 0; i < num_vertices; i++) {
        atomic_init(&labels[i], i);
        /* Progress for huge graphs */
        if (num_vertices > 10000000 && i > 0 && i % 10000000 == 0) {
            fprintf(stderr, "  %d / %d (%.1f%%)\n", i, num_vertices, 100.0 * i / num_vertices);
            fflush(stderr);
        }
    }
    fprintf(stderr, "Labels initialized.\n");
    fflush(stderr);

    /* Setup chunking for work-stealing */
    int32_t chunk_size = num_vertices / (num_threads * 4);
    if (chunk_size < 512)
        chunk_size = 512;
    if (chunk_size > 16384) /* Larger chunks for huge graphs */
        chunk_size = 16384;

    const int32_t num_chunks = (num_vertices + chunk_size - 1) / chunk_size;

    /* For huge graphs, disable subdivision to reduce overhead */
    const bool disable_subdivision = (num_vertices > 1000000);
    if (disable_subdivision) {
        fprintf(stderr, "Large graph detected (%d vertices) - disabling subdivision\n",
                num_vertices);
        fprintf(stderr, "Using %d chunks of size %d\n", num_chunks, chunk_size);
        fflush(stderr);
    }

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

    /* Allocate subdivision task pools (per-worker, eliminates malloc) */
    const size_t subdivision_pool_size = (size_t)num_threads * MAX_SUBDIVISIONS_PER_WORKER;
    Task *subdivision_pool = calloc(subdivision_pool_size, sizeof(Task));
    _Atomic(int32_t) *worker_counters = calloc((size_t)num_threads, sizeof(_Atomic(int32_t)));

    if (subdivision_pool == NULL || worker_counters == NULL) {
        fprintf(stderr, "Error: Failed to allocate subdivision pools\n");
        free(labels);
        free(ctx);
        free(task_pool);
        free(subdivision_pool);
        free(worker_counters);
        /* iteration_counter not allocated yet */
        return NULL;
    }

    /* Initialize worker allocation counters */
    for (int32_t i = 0; i < num_threads; i++) {
        atomic_init(&worker_counters[i], 0);
    }

    /* Create threadpool */
    const int64_t deque_capacity = num_chunks * 4;
    ThreadPool *pool = threadpool_create(num_threads, deque_capacity);
    if (pool == NULL) {
        fprintf(stderr, "Error: Failed to create threadpool\n");
        free(labels);
        free(ctx);
        free(task_pool);
        free(subdivision_pool);
        free(worker_counters);
        return NULL;
    }

    /* Setup context */
    ctx->g = g;
    ctx->labels = labels;
    ctx->changed = &changed;
    ctx->num_vertices = num_vertices;
    ctx->pool = pool;
    ctx->subdivision_pool = subdivision_pool;
    ctx->worker_alloc_counters = worker_counters;
    ctx->num_workers = num_threads;
    ctx->disable_subdivision = disable_subdivision;

    /* Initialize task pool */
    for (int32_t i = 0; i < num_chunks; i++) {
        task_pool[i].func = process_chunk_once;
        task_pool[i].start_vertex = i * chunk_size;
        task_pool[i].end_vertex = (i == num_chunks - 1) ? num_vertices : (i + 1) * chunk_size;
        task_pool[i].context = ctx;
        task_pool[i].should_free = false;
    }

    /* Pre-populate worker deques with affinity-based distribution
     * Each worker gets every Nth chunk (round-robin) BEFORE workers start */
    for (int32_t worker_id = 0; worker_id < num_threads; worker_id++) {
        Worker *worker = &pool->workers[worker_id];
        int32_t worker_chunks = 0;

        /* Assign chunks in round-robin fashion to this worker */
        for (int32_t chunk_id = worker_id; chunk_id < num_chunks; chunk_id += num_threads) {
            /* Increment active tasks before pushing */
            atomic_fetch_add_explicit(&pool->active_tasks, 1, memory_order_acq_rel);

            if (!deque_push_bottom(&worker->deque, &task_pool[chunk_id])) {
                /* This shouldn't happen with proper deque sizing, but handle it */
                atomic_fetch_sub_explicit(&pool->active_tasks, 1, memory_order_acq_rel);
                fprintf(stderr, "ERROR: Failed to pre-populate worker %d deque (chunk %d)\n",
                        worker_id, chunk_id);
                threadpool_destroy(pool);
                free(labels);
                free(subdivision_pool);
                free(worker_counters);
                free(ctx);
                free(task_pool);
                return NULL;
            }
            worker_chunks++;
        }
    }

    /* Start workers */
    threadpool_start(pool);

    /* Simple iteration loop */
    int32_t num_iterations = 0;
    const int32_t MAX_ITERATIONS = 1000;

    while (num_iterations < MAX_ITERATIONS) {
        /* Reset changed flag before iteration */
        atomic_store_explicit(&changed, 0, memory_order_relaxed);

        /* Re-populate deques (after first iteration) */
        if (num_iterations > 0) {
            for (int32_t chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
                int32_t worker_id = chunk_id % num_threads;
                Worker *worker = &pool->workers[worker_id];

                atomic_fetch_add_explicit(&pool->active_tasks, 1, memory_order_acq_rel);

                if (!deque_push_bottom(&worker->deque, &task_pool[chunk_id])) {
                    atomic_fetch_sub_explicit(&pool->active_tasks, 1, memory_order_acq_rel);
                    fprintf(stderr, "ERROR: Deque full (chunk %d, iteration %d)\n",
                            chunk_id, num_iterations);
                    goto cleanup_and_exit;
                }
            }
        }

        /* Wait for completion */
        threadpool_wait(pool);

        /* Reset subdivision pools for next iteration */
        for (int32_t i = 0; i < num_threads; i++) {
            atomic_store_explicit(&ctx->worker_alloc_counters[i], 0, memory_order_relaxed);
        }

        num_iterations++;

        /* Check convergence */
        if (atomic_load_explicit(&changed, memory_order_acquire) == 0) {
            break;
        }
    }

cleanup_and_exit:

    /* Shutdown */
    threadpool_shutdown(pool);
    threadpool_destroy(pool);

    /* Copy to regular array */
    int32_t *final_labels = malloc(sizeof(int32_t) * (size_t)num_vertices);
    if (final_labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate final labels\n");
        free(labels);
        free(subdivision_pool);
        free(worker_counters);
        free(ctx);
        free(task_pool);
        return NULL;
    }

    for (int32_t i = 0; i < num_vertices; i++) {
        final_labels[i] = atomic_load_explicit(&labels[i], memory_order_relaxed);
    }

    /* Cleanup */
    free(labels);
    free(subdivision_pool);
    free(worker_counters);
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