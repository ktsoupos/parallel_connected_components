#include "cc_pthreads.h"
#include "cc_common.h"
#include "definitions.h"
#include "deque.h"
#include "threadpool.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct AfforestContext {
    const Graph *graph;
    int32_t *parents;  // Using regular int32_t, atomic ops via __atomic builtins
    Task *task_pool;
    int32_t num_tasks;
    int32_t neighbor_round;
} AfforestContext;

// Link function from GAP Benchmark Suite - lock-free union-find
static inline void link_vertices(int32_t u, int32_t v, int32_t *restrict parents) {
    int32_t p1 = __atomic_load_n(&parents[u], __ATOMIC_RELAXED);
    int32_t p2 = __atomic_load_n(&parents[v], __ATOMIC_RELAXED);

    while (p1 != p2) {
        const int32_t high = (p1 > p2) ? p1 : p2;
        const int32_t low = (p1 < p2) ? p1 : p2;
        const int32_t p_high = __atomic_load_n(&parents[high], __ATOMIC_RELAXED);
        int32_t expected = high;

        if ((p_high == low) ||  // Already 'low'
            (p_high == high && __atomic_compare_exchange_n(
                &parents[high], &expected, low, false,
                __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))) {
            break;
        }

        p1 = __atomic_load_n(&parents[expected], __ATOMIC_RELAXED);  // Update with actual value after CAS
        p2 = __atomic_load_n(&parents[low], __ATOMIC_RELAXED);
    }
}

// Sampling phase: link first 'neighbor_round' neighbors
static void process_sampling_chunk(Task *task) {
    if (task == NULL) return;

    AfforestContext *ctx = (AfforestContext *)task->context;
    const Graph *g = ctx->graph;
    const int32_t round = ctx->neighbor_round;

    for (int32_t u = task->start_vertex; u < task->end_vertex; u++) {
        int32_t num_neighbors = 0;
        const int32_t *neighbors = graph_get_neighbors(g, u, &num_neighbors);

        if (neighbors != NULL && round < num_neighbors) {
            const int32_t v = neighbors[round];
            link_vertices(u, v, ctx->parents);
        }
    }
}

// Compress phase: path compression
static void process_compress_chunk(Task *task) {
    if (task == NULL) return;

    AfforestContext *ctx = (AfforestContext *)task->context;
    int32_t *parents = ctx->parents;

    for (int32_t n = task->start_vertex; n < task->end_vertex; n++) {
        int32_t parent = __atomic_load_n(&parents[n], __ATOMIC_RELAXED);
        int32_t grandparent = __atomic_load_n(&parents[parent], __ATOMIC_RELAXED);

        while (grandparent != parent) {
            __atomic_store_n(&parents[n], grandparent, __ATOMIC_RELAXED);
            parent = grandparent;
            grandparent = __atomic_load_n(&parents[parent], __ATOMIC_RELAXED);
        }
    }
}

// Final linking phase: link remaining neighbors (skip largest component)
typedef struct FinalLinkContext {
    const Graph *graph;
    int32_t *parents;
    int32_t largest_component;
    int32_t neighbor_rounds;
    Task *task_pool;
    int32_t num_tasks;
} FinalLinkContext;

static void process_final_link_chunk(Task *task) {
    if (task == NULL) return;

    FinalLinkContext *ctx = (FinalLinkContext *)task->context;
    const Graph *g = ctx->graph;

    for (int32_t u = task->start_vertex; u < task->end_vertex; u++) {
        // Skip if in largest component
        int32_t parent_u = __atomic_load_n(&ctx->parents[u], __ATOMIC_RELAXED);
        if (parent_u == ctx->largest_component) continue;

        int32_t num_neighbors = 0;
        const int32_t *neighbors = graph_get_neighbors(g, u, &num_neighbors);

        // Process remaining neighbors (after sampling rounds)
        for (int32_t j = ctx->neighbor_rounds; j < num_neighbors; j++) {
            const int32_t v = neighbors[j];
            link_vertices(u, v, ctx->parents);
        }
    }
}

// Sample frequent element (find largest component)
static int32_t sample_frequent_element(const int32_t *comp, int32_t num_vertices, int32_t num_samples) {
    int32_t *sample_counts = calloc((size_t)num_vertices, sizeof(int32_t));
    if (sample_counts == NULL) return 0;

    unsigned int seed = (unsigned int)time(NULL);
    for (int32_t i = 0; i < num_samples; i++) {
        const int32_t idx = (int32_t)(rand_r(&seed) % (unsigned int)num_vertices);
        const int32_t component_id = comp[idx];
        if (component_id >= 0 && component_id < num_vertices) {
            sample_counts[component_id]++;
        }
    }

    int32_t most_frequent = 0;
    int32_t max_count = 0;
    for (int32_t i = 0; i < num_vertices; i++) {
        if (sample_counts[i] > max_count) {
            max_count = sample_counts[i];
            most_frequent = i;
        }
    }

    free(sample_counts);
    return most_frequent;
}

CCResult *afforest_pthreads(const Graph *g, int32_t num_threads, int32_t neighbor_rounds) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    const int32_t num_vertices = g->num_vertices;

    // Allocate parent array
    int32_t *parents = aligned_alloc(64, sizeof(int32_t) * (size_t)num_vertices);
    if (parents == NULL) {
        fprintf(stderr, "Error: Failed to allocate parent array\n");
        return NULL;
    }

    // Initialize: each vertex is its own parent
    for (int32_t i = 0; i < num_vertices; i++) {
        parents[i] = i;
    }

    // Default neighbor rounds
    if (neighbor_rounds <= 0) {
        neighbor_rounds = 2;
    }

    // Create context
    AfforestContext *context = malloc(sizeof(AfforestContext));
    if (context == NULL) {
        fprintf(stderr, "Error: Failed to allocate context\n");
        free(parents);
        return NULL;
    }

    context->graph = g;
    context->parents = parents;

    // Determine chunk size
    int32_t chunk_size = (num_vertices / (num_threads * 2)) + 1;
    if (chunk_size < 4096) chunk_size = 4096;
    if (chunk_size > 65536) chunk_size = 65536;

    const int32_t num_chunks = (num_vertices + chunk_size - 1) / chunk_size;
    context->num_tasks = num_chunks;

    // Create task pool
    context->task_pool = calloc((size_t)num_chunks, sizeof(Task));
    if (context->task_pool == NULL) {
        fprintf(stderr, "Error: Failed to allocate task pool\n");
        free(parents);
        free(context);
        return NULL;
    }

    // Initialize tasks
    for (int32_t i = 0; i < num_chunks; i++) {
        Task *task = &context->task_pool[i];
        task->start_vertex = i * chunk_size;
        task->end_vertex = task->start_vertex + chunk_size;
        if (task->end_vertex > num_vertices) {
            task->end_vertex = num_vertices;
        }
        task->context = (void *)context;
        task->should_free = false;
    }

    // Create threadpool
    const int64_t deque_capacity = (num_chunks / num_threads) + 64;
    ThreadPool *pool = threadpool_create(num_threads, deque_capacity);
    if (pool == NULL) {
        fprintf(stderr, "Error: Failed to create threadpool\n");
        free(context->task_pool);
        free(parents);
        free(context);
        return NULL;
    }

    threadpool_start(pool);

    // === SAMPLING PHASE: Process first N neighbors ===
    for (int32_t round = 0; round < neighbor_rounds; round++) {
        context->neighbor_round = round;

        // Submit sampling tasks
        for (int32_t i = 0; i < num_chunks; i++) {
            Task *task = &context->task_pool[i];
            task->func = process_sampling_chunk;

            const int32_t worker_id = i % num_threads;
            Worker *worker = &pool->workers[worker_id];

            if (!deque_push_bottom(&worker->deque, task)) {
                fprintf(stderr, "Warning: Failed to push sampling task\n");
                continue;
            }
            atomic_fetch_add_explicit(&pool->active_tasks, 1, memory_order_release);
        }

        threadpool_wake_workers(pool);
        threadpool_barrier(pool);

        // Compression after each round
        for (int32_t i = 0; i < num_chunks; i++) {
            Task *task = &context->task_pool[i];
            task->func = process_compress_chunk;

            const int32_t worker_id = i % num_threads;
            Worker *worker = &pool->workers[worker_id];

            if (!deque_push_bottom(&worker->deque, task)) {
                fprintf(stderr, "Warning: Failed to push compress task\n");
                continue;
            }
            atomic_fetch_add_explicit(&pool->active_tasks, 1, memory_order_release);
        }

        threadpool_wake_workers(pool);
        threadpool_barrier(pool);
    }

    // === IDENTIFY LARGEST COMPONENT ===
    const int32_t largest_component = sample_frequent_element(parents, num_vertices, 1024);

    // === FINAL LINKING PHASE (skip largest component) ===
    FinalLinkContext *final_ctx = malloc(sizeof(FinalLinkContext));
    if (final_ctx == NULL) {
        fprintf(stderr, "Error: Failed to allocate final context\n");
        threadpool_shutdown(pool);
        threadpool_destroy(pool);
        free(context->task_pool);
        free(parents);
        free(context);
        return NULL;
    }

    final_ctx->graph = g;
    final_ctx->parents = parents;
    final_ctx->largest_component = largest_component;
    final_ctx->neighbor_rounds = neighbor_rounds;
    final_ctx->num_tasks = num_chunks;
    final_ctx->task_pool = context->task_pool;  // Reuse task pool

    // Update task contexts for final phase
    for (int32_t i = 0; i < num_chunks; i++) {
        Task *task = &context->task_pool[i];
        task->func = process_final_link_chunk;
        task->context = (void *)final_ctx;
    }

    // Submit final linking tasks
    for (int32_t i = 0; i < num_chunks; i++) {
        Task *task = &context->task_pool[i];

        const int32_t worker_id = i % num_threads;
        Worker *worker = &pool->workers[worker_id];

        if (!deque_push_bottom(&worker->deque, task)) {
            fprintf(stderr, "Warning: Failed to push final link task\n");
            continue;
        }
        atomic_fetch_add_explicit(&pool->active_tasks, 1, memory_order_release);
    }

    threadpool_wake_workers(pool);
    threadpool_barrier(pool);

    // Final compression
    for (int32_t i = 0; i < num_chunks; i++) {
        Task *task = &context->task_pool[i];
        task->func = process_compress_chunk;
        task->context = (void *)context;  // Back to original context

        const int32_t worker_id = i % num_threads;
        Worker *worker = &pool->workers[worker_id];

        if (!deque_push_bottom(&worker->deque, task)) {
            fprintf(stderr, "Warning: Failed to push final compress task\n");
            continue;
        }
        atomic_fetch_add_explicit(&pool->active_tasks, 1, memory_order_release);
    }

    threadpool_wake_workers(pool);
    threadpool_barrier(pool);

    threadpool_shutdown(pool);
    threadpool_destroy(pool);

    // Cleanup
    free(final_ctx);
    free(context->task_pool);
    free(context);

    // Create result
    CCResult *result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate result\n");
        free(parents);
        return NULL;
    }

    result->labels = parents;
    result->num_iterations = neighbor_rounds + 1;
    result->num_components = count_unique_labels(parents, num_vertices);

    return result;
}
