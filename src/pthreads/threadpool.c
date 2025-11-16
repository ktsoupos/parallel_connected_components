#include "threadpool.h"

#include "worker.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

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

    for (int i = 0; i < num_workers; i++) {
        worker_init(&pool->workers[i], i, deque_capacity);
        pool->workers[i].pool = pool; // Set backpointer
    }

    // 6. Initialize atomics
    atomic_init(&pool->active_tasks, 0);
    atomic_init(&pool->shutdown, false);

    // 7. Initialize barrier (all workers)
    pthread_barrier_init(&pool->start_barrier, NULL, (uint32_t)num_workers);
    pthread_mutex_init(&pool->tasks_done_mutex, NULL);
    pthread_cond_init(&pool->tasks_done_cond, NULL);

    return pool;
}

void threadpool_start(ThreadPool *pool) {
    if (pool == NULL) {
        return;
    }
    for (int32_t i = 0; i < pool->num_workers; i++) {
        const int res = pthread_create(
            &pool->threads[i],
            NULL,
            worker_thread_func,
            &pool->workers[i]);
        if (res != 0) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            // Handle error - maybe set shutdown flag?
        }
    }
}

void threadpool_wait(ThreadPool *pool) {
    if (!pool)
        return;

    while (atomic_load(&pool->active_tasks) > 0) {
        pthread_mutex_lock(&pool->tasks_done_mutex);

        if (atomic_load(&pool->active_tasks) > 0) {
            pthread_cond_wait(&pool->tasks_done_cond,
                              &pool->tasks_done_mutex);
        }

        pthread_mutex_unlock(&pool->tasks_done_mutex);
    }

    atomic_store(&pool->shutdown, 1);

    for (int i = 0; i < pool->num_workers; i++) {
        pthread_join(pool->threads[i], NULL);
    }
}

void threadpool_shutdown(ThreadPool *pool) {
    if (pool == NULL) {
        return;
    }

    // Signal all workers to stop
    atomic_store(&pool->shutdown, 1);

    // Broadcast to wake up any waiting threads
    pthread_mutex_lock(&pool->tasks_done_mutex);
    pthread_cond_broadcast(&pool->tasks_done_cond);
    pthread_mutex_unlock(&pool->tasks_done_mutex);

    // Join all threads
    for (int i = 0; i < pool->num_workers; i++) {
        pthread_join(pool->threads[i], NULL);
    }
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
    pthread_mutex_destroy(&pool->tasks_done_mutex);
    pthread_cond_destroy(&pool->tasks_done_cond);

    // Free arrays
    free(pool->workers);
    free(pool->threads);

    // Free pool itself
    free(pool);
}
