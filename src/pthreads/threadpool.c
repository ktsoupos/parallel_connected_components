#ifdef __linux__
#    define _GNU_SOURCE
#endif

#include "threadpool.h"

#include "worker.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __linux__
#    include <sched.h>
#    include <unistd.h>
#endif

ThreadPool *threadpool_create(int32_t num_workers, int64_t deque_capacity) {
    if (num_workers <= 0 || deque_capacity <= 0) {
        return NULL;
    }
    ThreadPool *pool = malloc(sizeof(ThreadPool));
    if (!pool) {
        return NULL;
    }
    pool->workers = calloc((uint32_t)num_workers, sizeof(Worker));
    if (!pool->workers) {
        free(pool);
        return NULL;
    }
    pool->threads = calloc((uint32_t)num_workers, sizeof(pthread_t));
    if (!pool->threads) {
        free(pool->workers);
        free(pool);
        return NULL;
    }
    pool->num_workers = num_workers;

    // Get number of CPU cores for affinity
#ifdef __linux__
    pool->num_numa_nodes = (int32_t)sysconf(_SC_NPROCESSORS_ONLN);
    pool->numa_available = true;
#    ifdef DEBUG
    fprintf(stderr, "[CPU Affinity] %d cores detected, will pin %d workers\n", pool->num_numa_nodes,
            num_workers);
#    endif
#else
    pool->num_numa_nodes = 1;
    pool->numa_available = false;
#endif

    for (int i = 0; i < num_workers; i++) {
        worker_init(&pool->workers[i], i, deque_capacity);
        pool->workers[i].pool = pool; // Set backpointer
        // Assign worker to CPU core (one worker per core, round-robin if more workers than cores)
        pool->workers[i].numa_node = pool->numa_available ? (i % pool->num_numa_nodes) : 0;
    }

    // 6. Initialize atomics
    atomic_init(&pool->active_tasks, 0);
    atomic_init(&pool->shutdown, false);
    atomic_init(&pool->barrier_waiting, true); // Start in barrier mode

    // 7. Initialize barriers
    pthread_barrier_init(&pool->start_barrier, NULL, (uint32_t)num_workers);
    pthread_barrier_init(&pool->iter_barrier, NULL, (uint32_t)(num_workers + 1)); // +1 for main
    pthread_mutex_init(&pool->tasks_done_mutex, NULL);
    pthread_cond_init(&pool->tasks_done_cond, NULL);
    pthread_mutex_init(&pool->work_mutex, NULL);
    pthread_cond_init(&pool->work_available, NULL);

    return pool;
}

void threadpool_start(ThreadPool *pool) {
    if (pool == NULL) {
        return;
    }
    for (int32_t i = 0; i < pool->num_workers; i++) {
        const int res =
            pthread_create(&pool->threads[i], NULL, worker_thread_func, &pool->workers[i]);
        if (res != 0) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            // Handle error - maybe set shutdown flag?
        }
    }
}

void threadpool_wait(ThreadPool *pool) {
    if (!pool)
        return;

    pthread_mutex_lock(&pool->tasks_done_mutex);
    while (atomic_load_explicit(&pool->active_tasks, memory_order_acquire) > 0) {
        pthread_cond_wait(&pool->tasks_done_cond, &pool->tasks_done_mutex);
    }
    pthread_mutex_unlock(&pool->tasks_done_mutex);
}

void threadpool_barrier(ThreadPool *pool) {
    if (!pool)
        return;

    // First, wait for all tasks to complete
    while (atomic_load(&pool->active_tasks) > 0) {
        pthread_mutex_lock(&pool->tasks_done_mutex);
        if (atomic_load(&pool->active_tasks) > 0) {
            pthread_cond_wait(&pool->tasks_done_cond, &pool->tasks_done_mutex);
        }
        pthread_mutex_unlock(&pool->tasks_done_mutex);
    }

    // Then synchronize with all workers at barrier
    // pthread_barrier_wait(&pool->iter_barrier);
}

void threadpool_wake_workers(ThreadPool *pool) {
    if (!pool)
        return;

    pthread_mutex_lock(&pool->tasks_done_mutex);
    pthread_cond_broadcast(&pool->tasks_done_cond);
    pthread_mutex_unlock(&pool->tasks_done_mutex);
}

void threadpool_shutdown(ThreadPool *pool) {
    atomic_store_explicit(&pool->shutdown, 1, memory_order_release);

    pthread_mutex_lock(&pool->work_mutex);
    pthread_cond_broadcast(&pool->work_available);
    pthread_mutex_unlock(&pool->work_mutex);

    for (int i = 0; i < pool->num_workers; i++)
        pthread_join(pool->threads[i], NULL);
}

void threadpool_destroy(ThreadPool *pool) {
    if (pool == NULL) {
        return;
    }

    // Cleanup each worker
    for (int i = 0; i < pool->num_workers; i++) {
        worker_cleanup(&pool->workers[i]);
    }

    // Destroy synchronization primitives
    pthread_barrier_destroy(&pool->start_barrier);
    pthread_barrier_destroy(&pool->iter_barrier);
    pthread_mutex_destroy(&pool->tasks_done_mutex);
    pthread_cond_destroy(&pool->tasks_done_cond);
    pthread_mutex_destroy(&pool->work_mutex);
    pthread_cond_destroy(&pool->work_available);

    // Free arrays
    free(pool->workers);
    free(pool->threads);

    // Free pool itself
    free(pool);
}