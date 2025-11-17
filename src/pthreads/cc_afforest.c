#include "cc_pthreads.h"
#include "cc_common.h"
#include "definitions.h"
#include "deque.h"
#include "threadpool.h"
#include "worker.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sched.h>
#include <unistd.h>

typedef struct AfforestContext {
    const Graph *graph;
    int32_t *parents;
    Task *task_pool;
    int32_t num_tasks;
    int32_t neighbor_round;
} AfforestContext;

typedef struct FinalLinkContext {
    const Graph *graph;
    int32_t *parents;
    int32_t largest_component;
    int32_t neighbor_rounds;
    Task *task_pool;
    int32_t num_tasks;
} FinalLinkContext;

/* Lock-free union-find link */
static inline void link_vertices(int32_t u, int32_t v, int32_t *restrict parents) {
    int retry = 0;
    int32_t p1 = __atomic_load_n(&parents[u], __ATOMIC_RELAXED);
    int32_t p2 = __atomic_load_n(&parents[v], __ATOMIC_RELAXED);

    while (p1 != p2) {
        int32_t high = (p1 > p2) ? p1 : p2;
        int32_t low = (p1 > p2) ? p2 : p1;
        int32_t p_high = __atomic_load_n(&parents[high], __ATOMIC_RELAXED);

        if (p_high == low)
            break;

        int32_t expected = p_high;
        if (__atomic_compare_exchange_n(&parents[high], &expected, low, false,
                                        __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
            break;

        if (++retry == 10000) {
            printf("[LINK-WARN][tid %lu] retry u=%d v=%d high=%d low=%d\n",
                   (unsigned long)pthread_self(), u, v, high, low);
            retry = 0;
        }

        p1 = __atomic_load_n(&parents[u], __ATOMIC_RELAXED);
        p2 = __atomic_load_n(&parents[v], __ATOMIC_RELAXED);
    }
}

/* Path compression */
static void process_compress_chunk(Task *task) {
    if (!task)
        return;
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

/* Sampling phase */
static void process_sampling_chunk(Task *task) {
    if (!task)
        return;
    AfforestContext *ctx = (AfforestContext *)task->context;
    const Graph *g = ctx->graph;
    int32_t round = ctx->neighbor_round;

    for (int32_t u = task->start_vertex; u < task->end_vertex; u++) {
        int32_t num_neighbors = 0;
        const int32_t *neighbors = graph_get_neighbors(g, u, &num_neighbors);
        if (neighbors && round < num_neighbors) {
            link_vertices(u, neighbors[round], ctx->parents);
        }
    }
}

/* Final linking phase */
static void process_final_link_chunk(Task *task) {
    if (!task)
        return;
    FinalLinkContext *ctx = (FinalLinkContext *)task->context;
    const Graph *g = ctx->graph;

    for (int32_t u = task->start_vertex; u < task->end_vertex; u++) {
        int32_t root_u = u;
        int32_t parent = __atomic_load_n(&ctx->parents[root_u], __ATOMIC_RELAXED);
        while (parent != root_u) {
            root_u = parent;
            parent = __atomic_load_n(&ctx->parents[root_u], __ATOMIC_RELAXED);
        }
        if (root_u == ctx->largest_component)
            continue;

        int32_t num_neighbors = 0;
        const int32_t *neighbors = graph_get_neighbors(g, u, &num_neighbors);
        for (int32_t j = ctx->neighbor_rounds; j < num_neighbors; j++) {
            link_vertices(u, neighbors[j], ctx->parents);
        }
    }
}

/* Sample largest component */
static int32_t
sample_frequent_element(int32_t *parents, int32_t num_vertices, int32_t num_samples) {
    int32_t *counts = calloc((size_t)num_vertices, sizeof(int32_t));
    if (!counts)
        return 0;

    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)(uintptr_t)parents;
    for (int32_t i = 0; i < num_samples; i++) {
        int32_t idx = (int32_t)(rand_r(&seed) % num_vertices);
        int32_t root = idx;
        int32_t p = __atomic_load_n(&parents[root], __ATOMIC_RELAXED);
        while (p != root) {
            root = p;
            p = __atomic_load_n(&parents[root], __ATOMIC_RELAXED);
        }
        counts[root]++;
    }

    int32_t max_idx = 0;
    int32_t max_count = 0;
    for (int32_t i = 0; i < num_vertices; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            max_idx = i;
        }
    }
    free(counts);
    return max_idx;
}

/* ------------------------ MAIN AFFOREST ------------------------ */
CCResult *afforest_pthreads(const Graph *g, int32_t num_threads, int32_t neighbor_rounds) {

    printf("[DEBUG] afforest_pthreads(): started\n");

    if (!g)
        return NULL;
    const int32_t num_vertices = g->num_vertices;
    printf("[DEBUG] num_vertices = %d\n", num_vertices);

    // Allocate parent array
    int32_t *parents = aligned_alloc(64, sizeof(int32_t) * (size_t)num_vertices);
    for (int32_t i = 0; i < num_vertices; i++)
        parents[i] = i;

    if (neighbor_rounds <= 0)
        neighbor_rounds = 2;
    printf("[DEBUG] Using %d neighbor rounds\n", neighbor_rounds);

    // Afforest context
    AfforestContext *context = malloc(sizeof(AfforestContext));
    context->graph = g;
    context->parents = parents;

    // Task chunk setup
    int32_t chunk_size = (num_vertices / (num_threads * 2)) + 1;
    if (chunk_size < 4096)
        chunk_size = 4096;
    if (chunk_size > 65536)
        chunk_size = 65536;
    const int32_t num_chunks = (num_vertices + chunk_size - 1) / chunk_size;
    context->num_tasks = num_chunks;

    context->task_pool = calloc((size_t)num_chunks, sizeof(Task));
    for (int32_t i = 0; i < num_chunks; i++) {
        Task *task = &context->task_pool[i];
        task->start_vertex = i * chunk_size;
        task->end_vertex = (task->start_vertex + chunk_size > num_vertices)
                               ? num_vertices
                               : task->start_vertex + chunk_size;
        task->context = context;
        task->should_free = false;
    }

    // ThreadPool
    printf("[DEBUG] Creating threadpool with %d threads\n", num_threads);
    ThreadPool *pool = threadpool_create(num_threads, (num_chunks / num_threads) + 64);
    threadpool_start(pool);
    printf("[DEBUG] Threadpool started\n");

    // ===========================
    // SAMPLING ROUNDS + COMPRESSION
    // ===========================
    for (int32_t round = 0; round < neighbor_rounds; round++) {
        printf("\n[DEBUG] === Sampling round %d ===\n", round);
        context->neighbor_round = round;

        for (int32_t i = 0; i < num_chunks; i++) {
            Task *task = &context->task_pool[i];
            task->func = process_sampling_chunk;

            int32_t worker_id = i % num_threads;
            printf("[DEBUG]  pushing sampling task chunk %d to worker %d\n", i, worker_id);

            deque_push_bottom(&pool->workers[worker_id].deque, task);
            atomic_fetch_add(&pool->active_tasks, 1);
        }

        // Wake workers and wait for completion
        threadpool_wake_workers(pool);
        threadpool_wait(pool);

        printf("[DEBUG] sampling round %d completed\n", round);

        // Compression
        printf("[DEBUG] starting compression after sampling round %d\n", round);
        for (int32_t i = 0; i < num_chunks; i++) {
            Task *task = &context->task_pool[i];
            task->func = process_compress_chunk;

            int32_t worker_id = i % num_threads;
            printf("[DEBUG] added task: %d\n", task->start_vertex);

            deque_push_bottom(&pool->workers[worker_id].deque, task);
            atomic_fetch_add(&pool->active_tasks, 1);
        }

        threadpool_wake_workers(pool);
        threadpool_wait(pool);

        printf("[DEBUG] compression phase done (post-round %d)\n", round);
    }

    // ===========================
    // IDENTIFY LARGEST COMPONENT
    // ===========================
    printf("\n[DEBUG] Finding largest component (sampling)\n");
    int32_t largest_component = sample_frequent_element(parents, num_vertices, 1024);
    printf("[DEBUG] Largest component root ≈ %d\n", largest_component);

    // ===========================
    // FINAL LINKING
    // ===========================
    FinalLinkContext *final_ctx = malloc(sizeof(FinalLinkContext));
    final_ctx->graph = g;
    final_ctx->parents = parents;
    final_ctx->largest_component = largest_component;
    final_ctx->neighbor_rounds = neighbor_rounds;
    final_ctx->num_tasks = num_chunks;
    final_ctx->task_pool = context->task_pool;

    printf("\n[DEBUG] === Starting final linking phase ===\n");
    for (int32_t i = 0; i < num_chunks; i++) {
        Task *task = &context->task_pool[i];
        task->func = process_final_link_chunk;
        task->context = final_ctx;

        int32_t worker_id = i % num_threads;
        printf("[DEBUG]  pushing final-link task %d -> worker %d\n", i, worker_id);

        deque_push_bottom(&pool->workers[worker_id].deque, task);
        atomic_fetch_add(&pool->active_tasks, 1);
    }

    threadpool_wake_workers(pool);
    threadpool_wait(pool);
    printf("[DEBUG] Final linking complete\n");

    // ===========================
    // FINAL COMPRESSION
    // ===========================
    printf("[DEBUG] Starting final compression\n");
    for (int32_t i = 0; i < num_chunks; i++) {
        Task *task = &context->task_pool[i];
        task->func = process_compress_chunk;
        task->context = context;

        int32_t worker_id = i % num_threads;
        deque_push_bottom(&pool->workers[worker_id].deque, task);
        atomic_fetch_add(&pool->active_tasks, 1);
    }

    threadpool_wake_workers(pool);
    threadpool_wait(pool);
    printf("[DEBUG] Final compression finished\n");

    // ===========================
    // Shutdown
    // ===========================
    threadpool_shutdown(pool);
    threadpool_destroy(pool);

    free(final_ctx);
    free(context->task_pool);
    free(context);

    CCResult *result = malloc(sizeof(CCResult));
    result->labels = parents;
    result->num_iterations = neighbor_rounds + 1;
    result->num_components = count_unique_labels(parents, num_vertices);

    printf("[DEBUG] afforest_pthreads(): DONE. Components = %d\n", result->num_components);
    return result;
}