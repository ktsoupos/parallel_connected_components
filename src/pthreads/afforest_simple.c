/**
 * Afforest Algorithm - Simple Pthreads Implementation
 * Converted from OpenMP version with basic pthread primitives
 */

#include "afforest_simple.h"
#include "cc_common.h"
#include "graph.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>

/* Thread argument structures */
typedef struct {
    int thread_id;
    int num_threads;
    int32_t *parents;
    int32_t num_vertices;
    int32_t start_idx;
    int32_t end_idx;
} compress_args_t;

typedef struct {
    int thread_id;
    int num_threads;
    const int32_t *comp;
    int32_t *sample_counts;
    int32_t num_vertices;
    int32_t num_samples;
    unsigned int seed;
} sample_args_t;

typedef struct {
    int thread_id;
    int num_threads;
    const int32_t *sample_counts;
    int32_t num_vertices;
    int32_t local_max_id;
    int32_t local_max_count;
} maxfind_args_t;

typedef struct {
    int thread_id;
    int num_threads;
    const int32_t *labels;
    bool *seen;
    int32_t num_vertices;
    int32_t local_count;
} count_labels_args_t;

typedef struct {
    int thread_id;
    int num_threads;
    const Graph *g;
    int32_t *parents;
    int32_t num_vertices;
    int32_t round;
} neighbor_round_args_t;

typedef struct {
    int thread_id;
    int num_threads;
    const Graph *g;
    int32_t *parents;
    int32_t num_vertices;
    int32_t neighbor_rounds;
    int32_t largest_component;
    int32_t start_idx;
    int32_t end_idx;
} final_phase_args_t;

typedef struct {
    int thread_id;
    int num_threads;
    int32_t *parents;
    int32_t start_idx;
    int32_t end_idx;
} init_args_t;

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

/**
 * Thread function for initialization
 */
static void *init_thread(void *arg) {
    init_args_t *args = (init_args_t *)arg;
    for (int32_t j = args->start_idx; j < args->end_idx; j++) {
        args->parents[j] = j;
    }
    return NULL;
}

/**
 * Thread function for compress operation
 */
static void *compress_thread(void *arg) {
    compress_args_t *args = (compress_args_t *)arg;

    /* Static partitioning with chunk size of 2048 */
    const int32_t chunk_size = 2048;
    const int32_t total_chunks = (args->num_vertices + chunk_size - 1) / chunk_size;

    for (int32_t chunk = args->thread_id; chunk < total_chunks; chunk += args->num_threads) {
        const int32_t start = chunk * chunk_size;
        const int32_t end = (start + chunk_size < args->num_vertices) ?
                            start + chunk_size : args->num_vertices;

        for (int32_t n = start; n < end; n++) {
            while (args->parents[args->parents[n]] != args->parents[n]) {
                args->parents[n] = args->parents[args->parents[n]];
            }
        }
    }

    return NULL;
}

static void compress(int32_t *restrict parents, int32_t num_vertices, int num_threads) {
    pthread_t *threads = malloc(sizeof(pthread_t) * (size_t)num_threads);
    compress_args_t *args = malloc(sizeof(compress_args_t) * (size_t)num_threads);

    if (threads == NULL || args == NULL) {
        fprintf(stderr, "Error: Failed to allocate thread resources\n");
        free(threads);
        free(args);
        return;
    }

    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].num_threads = num_threads;
        args[i].parents = parents;
        args[i].num_vertices = num_vertices;

        pthread_create(&threads[i], NULL, compress_thread, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(args);
}

/**
 * Thread function for sampling phase
 */
static void *sample_thread(void *arg) {
    sample_args_t *args = (sample_args_t *)arg;

    /* Static scheduling */
    const int32_t samples_per_thread = args->num_samples / args->num_threads;
    const int32_t start = args->thread_id * samples_per_thread;
    int32_t end = start + samples_per_thread;

    /* Last thread takes remaining samples */
    if (args->thread_id == args->num_threads - 1) {
        end = args->num_samples;
    }

    unsigned int thread_seed = args->seed;

    for (int32_t i = start; i < end; i++) {
        /* Thread-safe random number generation */
        const unsigned int rand_val = rand_r(&thread_seed);
        const int32_t idx = (int32_t)(rand_val % (unsigned int)args->num_vertices);
        const int32_t component_id = args->comp[idx];

        /* Bounds check to prevent heap corruption */
        if (component_id >= 0 && component_id < args->num_vertices) {
            /* Atomic increment to avoid race conditions */
            __atomic_fetch_add(&args->sample_counts[component_id], 1, __ATOMIC_SEQ_CST);
        }
    }

    return NULL;
}

/**
 * Thread function for max-finding phase
 */
static void *maxfind_thread(void *arg) {
    maxfind_args_t *args = (maxfind_args_t *)arg;

    /* Each thread finds local maximum */
    args->local_max_id = 0;
    args->local_max_count = 0;

    /* Static scheduling */
    const int32_t elems_per_thread = args->num_vertices / args->num_threads;
    const int32_t start = args->thread_id * elems_per_thread;
    int32_t end = start + elems_per_thread;

    /* Last thread takes remaining elements */
    if (args->thread_id == args->num_threads - 1) {
        end = args->num_vertices;
    }

    for (int32_t i = start; i < end; i++) {
        if (args->sample_counts[i] > args->local_max_count) {
            args->local_max_count = args->sample_counts[i];
            args->local_max_id = i;
        }
    }

    return NULL;
}

/**
 * Parallel version of sample_frequent_element using pthreads
 */
static int32_t sample_frequent_element(const int32_t *comp, const int32_t num_vertices,
                                 const int32_t num_samples, int num_threads) {
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

    pthread_t *threads = malloc(sizeof(pthread_t) * (size_t)num_threads);
    sample_args_t *sample_args = malloc(sizeof(sample_args_t) * (size_t)num_threads);

    if (threads == NULL || sample_args == NULL) {
        fprintf(stderr, "Error: Failed to allocate thread resources\n");
        free(sample_counts);
        free(threads);
        free(sample_args);
        return -1;
    }

    for (int i = 0; i < num_threads; i++) {
        sample_args[i].thread_id = i;
        sample_args[i].num_threads = num_threads;
        sample_args[i].comp = comp;
        sample_args[i].sample_counts = sample_counts;
        sample_args[i].num_vertices = num_vertices;
        sample_args[i].num_samples = num_samples;
        sample_args[i].seed = base_seed + (unsigned int)i;

        pthread_create(&threads[i], NULL, sample_thread, &sample_args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Parallel max-finding phase */
    maxfind_args_t *maxfind_args = malloc(sizeof(maxfind_args_t) * (size_t)num_threads);
    if (maxfind_args == NULL) {
        fprintf(stderr, "Error: Failed to allocate maxfind resources\n");
        free(sample_counts);
        free(threads);
        free(sample_args);
        return -1;
    }

    for (int i = 0; i < num_threads; i++) {
        maxfind_args[i].thread_id = i;
        maxfind_args[i].num_threads = num_threads;
        maxfind_args[i].sample_counts = sample_counts;
        maxfind_args[i].num_vertices = num_vertices;

        pthread_create(&threads[i], NULL, maxfind_thread, &maxfind_args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Combine thread-local results */
    int32_t most_frequent_id = 0;
    int32_t max_count = 0;

    for (int i = 0; i < num_threads; i++) {
        if (maxfind_args[i].local_max_count > max_count) {
            max_count = maxfind_args[i].local_max_count;
            most_frequent_id = maxfind_args[i].local_max_id;
        }
    }

    /* Calculate and print percentage */
    const float percentage = (float)max_count / (float)num_samples * 100.0f;
    printf("Skipping largest intermediate component (ID: %d, approx. %.1f%% of the graph)\n",
           most_frequent_id, percentage);

    free(sample_counts);
    free(threads);
    free(sample_args);
    free(maxfind_args);

    return most_frequent_id;
}

/**
 * Thread function for neighbor sampling rounds
 */
static void *neighbor_round_thread(void *arg) {
    neighbor_round_args_t *args = (neighbor_round_args_t *)arg;

    /* Static scheduling */
    const int32_t verts_per_thread = args->num_vertices / args->num_threads;
    const int32_t start = args->thread_id * verts_per_thread;
    int32_t end = start + verts_per_thread;

    /* Last thread takes remaining vertices */
    if (args->thread_id == args->num_threads - 1) {
        end = args->num_vertices;
    }

    for (int32_t u = start; u < end; u++) {
        int32_t num_neighbors = 0;
        const int32_t *neighbors = graph_get_neighbors(args->g, u, &num_neighbors);

        if (neighbors != NULL && args->round < num_neighbors) {
            const int32_t v = neighbors[args->round];
            link_vertices(u, v, args->parents);
        }
    }

    return NULL;
}

/**
 * Thread function for final phase (dynamic scheduling approximation)
 */
static void *final_phase_thread(void *arg) {
    final_phase_args_t *args = (final_phase_args_t *)arg;

    for (int32_t u = args->start_idx; u < args->end_idx; u++) {
        if (args->parents[u] == args->largest_component) continue;

        int32_t num_neighbors = 0;
        const int32_t *neighbors = graph_get_neighbors(args->g, u, &num_neighbors);

        /* Process remaining neighbors (after neighbor_rounds) */
        if (neighbors != NULL) {
            for (int32_t j = args->neighbor_rounds; j < num_neighbors; j++) {
                const int32_t v = neighbors[j];
                link_vertices(u, v, args->parents);
            }
        }
    }

    return NULL;
}

CCResult *afforest_simple_pthreads(const Graph *restrict g, int32_t num_threads, int32_t neighbor_rounds) {
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
    pthread_t *threads = malloc(sizeof(pthread_t) * (size_t)num_threads);
    init_args_t *init_args = malloc(sizeof(init_args_t) * (size_t)num_threads);

    if (threads == NULL || init_args == NULL) {
        fprintf(stderr, "Error: Failed to allocate thread resources\n");
        free(parents);
        free(result);
        free(threads);
        free(init_args);
        return NULL;
    }

    /* Parallel initialization */
    for (int i = 0; i < num_threads; i++) {
        const int32_t verts_per_thread = num_vertices / num_threads;
        const int32_t start = i * verts_per_thread;
        const int32_t end = (i == num_threads - 1) ? num_vertices : start + verts_per_thread;

        init_args[i].thread_id = i;
        init_args[i].num_threads = num_threads;
        init_args[i].start_idx = start;
        init_args[i].end_idx = end;
        init_args[i].parents = parents;

        pthread_create(&threads[i], NULL, init_thread, &init_args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Set default neighbor_rounds if needed */
    if (neighbor_rounds <= 0) {
        neighbor_rounds = 2; // Default: sample first 2 neighbors
    }

    /* Neighbor sampling rounds */
    neighbor_round_args_t *neighbor_args = malloc(sizeof(neighbor_round_args_t) * (size_t)num_threads);
    if (neighbor_args == NULL) {
        fprintf(stderr, "Error: Failed to allocate neighbor args\n");
        free(parents);
        free(result);
        free(threads);
        free(init_args);
        return NULL;
    }

    for (int32_t r = 0; r < neighbor_rounds; ++r) {
        for (int i = 0; i < num_threads; i++) {
            neighbor_args[i].thread_id = i;
            neighbor_args[i].num_threads = num_threads;
            neighbor_args[i].g = g;
            neighbor_args[i].parents = parents;
            neighbor_args[i].num_vertices = num_vertices;
            neighbor_args[i].round = r;

            pthread_create(&threads[i], NULL, neighbor_round_thread, &neighbor_args[i]);
        }

        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        compress(parents, num_vertices, num_threads);
    }

    const int32_t largest_component = sample_frequent_element(parents, num_vertices, 1024, num_threads);

    /* Final phase - process remaining neighbors */
    final_phase_args_t *final_args = malloc(sizeof(final_phase_args_t) * (size_t)num_threads);
    if (final_args == NULL) {
        fprintf(stderr, "Error: Failed to allocate final phase args\n");
        free(parents);
        free(result);
        free(threads);
        free(init_args);
        free(neighbor_args);
        return NULL;
    }

    /* Approximate dynamic scheduling by giving each thread a range */
    for (int i = 0; i < num_threads; i++) {
        const int32_t verts_per_thread = num_vertices / num_threads;
        const int32_t start = i * verts_per_thread;
        const int32_t end = (i == num_threads - 1) ? num_vertices : start + verts_per_thread;

        final_args[i].thread_id = i;
        final_args[i].num_threads = num_threads;
        final_args[i].g = g;
        final_args[i].parents = parents;
        final_args[i].num_vertices = num_vertices;
        final_args[i].neighbor_rounds = neighbor_rounds;
        final_args[i].largest_component = largest_component;
        final_args[i].start_idx = start;
        final_args[i].end_idx = end;

        pthread_create(&threads[i], NULL, final_phase_thread, &final_args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    compress(parents, num_vertices, num_threads);

    /* Store final labels in result */
    result->labels = parents;
    result->num_iterations = neighbor_rounds + 1; // Sampling rounds + final phase

    /* Count connected components */
    result->num_components = count_unique_labels(result->labels, num_vertices);
    if (result->num_components < 0) {
        fprintf(stderr, "Error: Failed to count components\n");
        free(result->labels);
        free(result);
        free(threads);
        free(init_args);
        free(neighbor_args);
        free(final_args);
        return NULL;
    }

    /* Cleanup */
    free(threads);
    free(init_args);
    free(neighbor_args);
    free(final_args);

    return result;
}
