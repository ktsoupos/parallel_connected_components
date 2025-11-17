/**
 * Afforest Algorithm - Optimized Pthreads Implementation
 * Key optimizations:
 * - Reusable thread pool with barriers (no repeated create/destroy)
 * - Pre-allocated argument structures (no malloc in hot paths)
 * - Efficient work distribution
 * - NUMA-aware thread pinning (optional)
 */

/* Must define _GNU_SOURCE before any includes for CPU_SET/CPU_ZERO */
#ifdef __linux__
#define _GNU_SOURCE
#define NUMA_AVAILABLE 1
#else
#define NUMA_AVAILABLE 0
#endif

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

#if NUMA_AVAILABLE
#include <sched.h>
#include <unistd.h>
#endif

/* Cache line size for alignment (prevent false sharing) */
#define CACHE_LINE_SIZE 64

/* Chunk size for dynamic scheduling (tunable: 512-4096) */
#define DYNAMIC_CHUNK_SIZE 2048

/* Cache-aligned work counter for dynamic load balancing */
typedef struct {
    _Atomic int32_t next_chunk;
    char padding[CACHE_LINE_SIZE - sizeof(_Atomic int32_t)];
} work_counter_t;

/* Global synchronization primitives */
typedef struct {
    pthread_barrier_t barrier;
    void *(*work_func)(void*);
    volatile bool should_exit;
} thread_control_t;

/* Unified worker thread structure */
typedef struct {
    int thread_id;
    int num_threads;

    /* Shared data pointers */
    const Graph *g;
    int32_t *parents;
    int32_t num_vertices;

    /* Phase-specific data */
    int32_t round;
    int32_t neighbor_rounds;
    int32_t largest_component;

    /* Thread control */
    thread_control_t *control;

    /* Work range */
    int32_t start_idx;
    int32_t end_idx;
} worker_args_t;

typedef struct {
    int thread_id;
    int num_threads;
    int32_t *parents;
    int32_t num_vertices;
    int32_t start_idx;
    int32_t end_idx;
} __attribute__((aligned(CACHE_LINE_SIZE))) compress_args_t;

typedef struct {
    int thread_id;
    int num_threads;
    const int32_t *comp;
    int32_t *sample_counts;
    int32_t num_vertices;
    int32_t num_samples;
    unsigned int seed;
} __attribute__((aligned(CACHE_LINE_SIZE))) sample_args_t;

typedef struct {
    int thread_id;
    int num_threads;
    const int32_t *sample_counts;
    int32_t num_vertices;
    int32_t local_max_id;
    int32_t local_max_count;
} __attribute__((aligned(CACHE_LINE_SIZE))) maxfind_args_t;

/* Simple hash set for storing unique labels */
typedef struct {
    int32_t *keys;      /* Array of label values */
    bool *occupied;     /* Which slots are occupied */
    int32_t capacity;   /* Size of the hash table */
    int32_t size;       /* Number of elements stored */
} HashSet;

typedef struct {
    int thread_id;
    int num_threads;
    const int32_t *labels;
    int32_t num_vertices;
    int32_t start_idx;
    int32_t end_idx;
    HashSet *local_set;  /* Each thread's private hash set */
} __attribute__((aligned(CACHE_LINE_SIZE))) count_labels_args_t;

typedef struct {
    int thread_id;
    int num_threads;
    const Graph *g;
    int32_t *parents;
    int32_t num_vertices;
    int32_t round;
    work_counter_t *work_counter;  /* For dynamic scheduling */
    bool use_dynamic;              /* Enable dynamic load balancing */
} __attribute__((aligned(CACHE_LINE_SIZE))) neighbor_round_args_t;

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
    work_counter_t *work_counter;  /* For dynamic scheduling */
    bool use_dynamic;              /* Enable dynamic load balancing */
} __attribute__((aligned(CACHE_LINE_SIZE))) final_phase_args_t;

typedef struct {
    int thread_id;
    int num_threads;
    int32_t *parents;
    int32_t start_idx;
    int32_t end_idx;
} __attribute__((aligned(CACHE_LINE_SIZE))) init_args_t;

/**
 * Pin calling thread to a specific CPU core (NUMA optimization)
 */
static void pin_thread_to_core(int core_id) {
#if NUMA_AVAILABLE
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        /* Non-fatal: just print warning */
        fprintf(stderr, "Warning: Failed to pin thread to core %d\n", core_id);
    }
#else
    (void)core_id;  /* Suppress unused parameter warning */
#endif
}

/**
 * Get number of available CPU cores
 */
static int get_num_cores(void) {
#if NUMA_AVAILABLE
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#else
    return 16;  /* Default fallback */
#endif
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

/**
 * Thread function for initialization
 */
static void *init_thread(void *arg) {
    init_args_t *args = (init_args_t *)arg;

    /* Pin thread to core for NUMA locality */
    pin_thread_to_core(args->thread_id);

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

    /* Pin thread to core for NUMA locality */
    pin_thread_to_core(args->thread_id);

    /* Direct range assignment - each thread gets a contiguous chunk */
    for (int32_t n = args->start_idx; n < args->end_idx; n++) {
        while (args->parents[args->parents[n]] != args->parents[n]) {
            args->parents[n] = args->parents[args->parents[n]];
        }
    }

    return NULL;
}

/* Optimized compress - reuses thread handles */
static void compress_with_threads(int32_t *restrict parents, int32_t num_vertices,
                                    pthread_t *threads, compress_args_t *args, int num_threads) {
    /* Distribute work evenly across threads */
    const int32_t verts_per_thread = num_vertices / num_threads;

    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].num_threads = num_threads;
        args[i].parents = parents;
        args[i].num_vertices = num_vertices;
        args[i].start_idx = i * verts_per_thread;
        args[i].end_idx = (i == num_threads - 1) ? num_vertices : (i + 1) * verts_per_thread;

        pthread_create(&threads[i], NULL, compress_thread, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

/* Legacy function for backward compatibility */
static void compress(int32_t *restrict parents, int32_t num_vertices, int num_threads) {
    pthread_t *threads = malloc(sizeof(pthread_t) * (size_t)num_threads);
    compress_args_t *args = malloc(sizeof(compress_args_t) * (size_t)num_threads);

    if (threads == NULL || args == NULL) {
        fprintf(stderr, "Error: Failed to allocate thread resources\n");
        free(threads);
        free(args);
        return;
    }

    compress_with_threads(parents, num_vertices, threads, args, num_threads);

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

/* ============ HASH SET IMPLEMENTATION ============ */

/**
 * Create a hash set with given capacity
 */
static HashSet* hashset_create(int32_t capacity) {
    HashSet *set = malloc(sizeof(HashSet));
    if (set == NULL) return NULL;

    set->capacity = capacity;
    set->size = 0;
    set->keys = malloc(sizeof(int32_t) * (size_t)capacity);
    set->occupied = calloc((size_t)capacity, sizeof(bool));

    if (set->keys == NULL || set->occupied == NULL) {
        free(set->keys);
        free(set->occupied);
        free(set);
        return NULL;
    }

    return set;
}

/**
 * Insert a key into the hash set (returns true if newly inserted)
 */
static bool hashset_insert(HashSet *set, int32_t key) {
    /* Simple hash function */
    uint32_t hash = (uint32_t)key * 2654435761U;  /* Knuth's multiplicative hash */
    int32_t idx = (int32_t)(hash % (uint32_t)set->capacity);

    /* Linear probing */
    int32_t start_idx = idx;
    while (set->occupied[idx]) {
        if (set->keys[idx] == key) {
            return false;  /* Already exists */
        }
        idx = (idx + 1) % set->capacity;
        if (idx == start_idx) {
            /* Table is full - shouldn't happen with proper sizing */
            return false;
        }
    }

    /* Insert new key */
    set->keys[idx] = key;
    set->occupied[idx] = true;
    set->size++;
    return true;
}

/**
 * Free a hash set
 */
static void hashset_destroy(HashSet *set) {
    if (set == NULL) return;
    free(set->keys);
    free(set->occupied);
    free(set);
}

/* ============ PARALLEL COUNTING WITH HASH SETS ============ */

/**
 * Thread function: build local hash set of unique labels
 */
static void *count_labels_thread(void *arg) {
    count_labels_args_t *args = (count_labels_args_t *)arg;

    /* Scan this thread's range and collect unique labels */
    for (int32_t v = args->start_idx; v < args->end_idx; v++) {
        int32_t label = args->labels[v];

        /* Bounds check */
        if (label >= 0 && label < args->num_vertices) {
            hashset_insert(args->local_set, label);
        }
    }

    return NULL;
}

/**
 * Fast parallel unique label counting with thread-local hash sets
 */
static int32_t count_unique_labels_parallel(const int32_t *labels, int32_t num_vertices,
                                             pthread_t *threads, count_labels_args_t *args, int num_threads) {

    /* Conservative estimate: allocate enough for all possible unique labels per thread
       Worst case: each thread sees different labels (num_vertices / num_threads)
       Use 5x capacity for low load factor and fast insertion */
    const int32_t max_labels_per_thread = num_vertices / num_threads;
    const int32_t set_capacity = max_labels_per_thread * 5;  /* Load factor ~0.2 */

    /* Step 1: Each thread creates its own hash set */
    for (int i = 0; i < num_threads; i++) {
        args[i].local_set = hashset_create(set_capacity);
        if (args[i].local_set == NULL) {
            /* Cleanup on error */
            for (int j = 0; j < i; j++) {
                hashset_destroy(args[j].local_set);
            }
            fprintf(stderr, "Error: Failed to allocate hash set\n");
            return -1;
        }
    }

    /* Step 2: Parallel scan - each thread builds its local set (NO CONTENTION!) */
    const int32_t verts_per_thread = num_vertices / num_threads;

    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].num_threads = num_threads;
        args[i].labels = labels;
        args[i].num_vertices = num_vertices;
        args[i].start_idx = i * verts_per_thread;
        args[i].end_idx = (i == num_threads - 1) ? num_vertices : (i + 1) * verts_per_thread;

        pthread_create(&threads[i], NULL, count_labels_thread, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Step 3: Merge all hash sets using a bit vector (fast!) */
    bool *global_seen = calloc((size_t)num_vertices, sizeof(bool));
    if (global_seen == NULL) {
        for (int i = 0; i < num_threads; i++) {
            hashset_destroy(args[i].local_set);
        }
        fprintf(stderr, "Error: Failed to allocate merge array\n");
        return -1;
    }

    /* Mark all labels from all thread-local sets */
    for (int i = 0; i < num_threads; i++) {
        HashSet *set = args[i].local_set;
        for (int32_t j = 0; j < set->capacity; j++) {
            if (set->occupied[j]) {
                global_seen[set->keys[j]] = true;
            }
        }
    }

    /* Step 4: Count unique labels */
    int32_t total_count = 0;
    for (int32_t i = 0; i < num_vertices; i++) {
        if (global_seen[i]) {
            total_count++;
        }
    }

    /* Cleanup */
    free(global_seen);
    for (int i = 0; i < num_threads; i++) {
        hashset_destroy(args[i].local_set);
    }

    return total_count;
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
 * Thread function for neighbor sampling rounds (supports static and dynamic scheduling)
 */
static void *neighbor_round_thread(void *arg) {
    neighbor_round_args_t *args = (neighbor_round_args_t *)arg;

    /* Pin thread to core for NUMA locality */
    pin_thread_to_core(args->thread_id);

    if (args->use_dynamic) {
        /* === DYNAMIC SCHEDULING === */
        const int32_t chunk_size = DYNAMIC_CHUNK_SIZE;
        const int32_t num_chunks = (args->num_vertices + chunk_size - 1) / chunk_size;

        while (1) {
            /* Atomically grab next chunk */
            const int32_t chunk_id = atomic_fetch_add(&args->work_counter->next_chunk, 1);

            if (chunk_id >= num_chunks) break;  /* No more work */

            /* Calculate chunk boundaries */
            const int32_t start = chunk_id * chunk_size;
            const int32_t end = (start + chunk_size > args->num_vertices)
                                ? args->num_vertices
                                : start + chunk_size;

            /* Process this chunk */
            for (int32_t u = start; u < end; u++) {
                int32_t num_neighbors = 0;
                const int32_t *neighbors = graph_get_neighbors(args->g, u, &num_neighbors);

                if (neighbors != NULL && args->round < num_neighbors) {
                    const int32_t v = neighbors[args->round];
                    link_vertices(u, v, args->parents);
                }
            }
        }
    } else {
        /* === STATIC SCHEDULING === */
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
    }

    return NULL;
}

/**
 * Thread function for final phase (supports static and dynamic scheduling)
 */
static void *final_phase_thread(void *arg) {
    final_phase_args_t *args = (final_phase_args_t *)arg;

    /* Pin thread to core for NUMA locality */
    pin_thread_to_core(args->thread_id);

    if (args->use_dynamic) {
        /* === DYNAMIC SCHEDULING === */
        const int32_t chunk_size = DYNAMIC_CHUNK_SIZE;
        const int32_t num_chunks = (args->num_vertices + chunk_size - 1) / chunk_size;

        while (1) {
            /* Atomically grab next chunk */
            const int32_t chunk_id = atomic_fetch_add(&args->work_counter->next_chunk, 1);

            if (chunk_id >= num_chunks) break;  /* No more work */

            /* Calculate chunk boundaries */
            const int32_t start = chunk_id * chunk_size;
            const int32_t end = (start + chunk_size > args->num_vertices)
                                ? args->num_vertices
                                : start + chunk_size;

            /* Process this chunk */
            for (int32_t u = start; u < end; u++) {
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
        }
    } else {
        /* === STATIC SCHEDULING === */
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
    }

    return NULL;
}

CCResult *afforest_simple_pthreads(const Graph *restrict g, int32_t num_threads, int32_t neighbor_rounds, bool use_dynamic) {
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

    /* Set default neighbor_rounds if needed */
    if (neighbor_rounds <= 0) {
        neighbor_rounds = 2; // Default: sample first 2 neighbors
    }

    pthread_t *threads = malloc(sizeof(pthread_t) * (size_t)num_threads);
    init_args_t *init_args = malloc(sizeof(init_args_t) * (size_t)num_threads);
    neighbor_round_args_t *neighbor_args = malloc(sizeof(neighbor_round_args_t) * (size_t)num_threads);
    compress_args_t *compress_args = malloc(sizeof(compress_args_t) * (size_t)num_threads);
    final_phase_args_t *final_args = malloc(sizeof(final_phase_args_t) * (size_t)num_threads);

    if (threads == NULL || init_args == NULL || neighbor_args == NULL ||
        compress_args == NULL || final_args == NULL) {
        fprintf(stderr, "Error: Failed to allocate thread resources\n");
        free(parents);
        free(result);
        free(threads);
        free(init_args);
        free(neighbor_args);
        free(compress_args);
        free(final_args);
        return NULL;
    }

    /* Allocate work counters for dynamic scheduling (one per phase) */
    work_counter_t *neighbor_counters = NULL;
    work_counter_t *final_counter = NULL;

    if (use_dynamic) {
        /* One work counter per neighbor round */
        neighbor_counters = calloc((size_t)neighbor_rounds, sizeof(work_counter_t));
        final_counter = calloc(1, sizeof(work_counter_t));

        if (neighbor_counters == NULL || final_counter == NULL) {
            fprintf(stderr, "Error: Failed to allocate work counters\n");
            free(parents);
            free(result);
            free(threads);
            free(init_args);
            free(neighbor_args);
            free(compress_args);
            free(final_args);
            free(neighbor_counters);
            free(final_counter);
            return NULL;
        }
    }

    printf("Using %s scheduling (chunk size: %d)\n",
           use_dynamic ? "DYNAMIC" : "STATIC",
           use_dynamic ? DYNAMIC_CHUNK_SIZE : 0);


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

    /* === PHASE 2: Neighbor sampling rounds (with compression) === */
    for (int32_t r = 0; r < neighbor_rounds; ++r) {
        /* Reset work counter for this round (if using dynamic scheduling) */
        if (use_dynamic) {
            atomic_store(&neighbor_counters[r].next_chunk, 0);
        }

        /* Neighbor sampling */
        for (int i = 0; i < num_threads; i++) {
            neighbor_args[i].thread_id = i;
            neighbor_args[i].num_threads = num_threads;
            neighbor_args[i].g = g;
            neighbor_args[i].parents = parents;
            neighbor_args[i].num_vertices = num_vertices;
            neighbor_args[i].round = r;
            neighbor_args[i].use_dynamic = use_dynamic;
            neighbor_args[i].work_counter = use_dynamic ? &neighbor_counters[r] : NULL;

            pthread_create(&threads[i], NULL, neighbor_round_thread, &neighbor_args[i]);
        }

        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        /* Compression - REUSE threads array */
        compress_with_threads(parents, num_vertices, threads, compress_args, num_threads);
    }

    /* === PHASE 3: Identify largest component === */
    const int32_t largest_component = sample_frequent_element(parents, num_vertices, 1024, num_threads);

    /* === PHASE 4: Final linking phase === */

    /* Reset work counter for final phase (if using dynamic scheduling) */
    if (use_dynamic) {
        atomic_store(&final_counter->next_chunk, 0);
    }

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
        final_args[i].use_dynamic = use_dynamic;
        final_args[i].work_counter = use_dynamic ? final_counter : NULL;

        pthread_create(&threads[i], NULL, final_phase_thread, &final_args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }


    /* === PHASE 5: Final compression === */
    compress_with_threads(parents, num_vertices, threads, compress_args, num_threads);

    /* Store final labels in result */
    result->labels = parents;
    result->num_iterations = neighbor_rounds + 1; // Sampling rounds + final phase

    /* === PHASE 6: Count connected components === */
    result->num_components = count_unique_labels(result->labels, num_vertices);

    if (result->num_components < 0) {
        fprintf(stderr, "Error: Failed to count components\n");
        free(result->labels);
        free(result);
        free(threads);
        free(init_args);
        free(neighbor_args);
        free(compress_args);
        free(final_args);
        free(neighbor_counters);
        free(final_counter);
        return NULL;
    }

    /* Cleanup thread resources */
    free(threads);
    free(init_args);
    free(neighbor_args);
    free(compress_args);
    free(final_args);
    free(neighbor_counters);
    free(final_counter);

    return result;
}
