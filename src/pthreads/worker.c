#include "worker.h"
#include "deque.h"
#include "utils.h"

#include <sched.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>


#define NUM_STEAL_MAX_ATTEMPTS 3

// Thread-local storage: each thread knows which worker it is
static __thread Worker *current_worker = NULL;
static __thread ThreadPool *current_pool = NULL;


// XORShift64 - fast, simple PRNG for per-worker random numbers
static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

void worker_init(Worker *worker, int32_t id, int64_t deque_capacity) {
    if (worker == NULL || deque_capacity <= 0) {
        return;
    }

    worker->id = id;
    worker->stop_flag = 0;
    worker->victim = -1;

    // Initialize RNG state (must be non-zero)
    // Mix worker ID with time to get unique seed per worker
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    worker->rng_state = ((uint64_t)id + 1) * 0x9e3779b97f4a7c15ULL + (uint64_t)ts.tv_nsec;
    if (worker->rng_state == 0) {
        worker->rng_state = 1; // XORShift requires non-zero state
    }

    // Round up to next power of 2 for efficient masking
    int64_t actual_capacity = next_power_of_two(deque_capacity);

    worker->deque.capacity = actual_capacity;
    worker->deque.mask = actual_capacity - 1;

    worker->deque.buffer = (Task **)calloc((size_t)actual_capacity, sizeof(Task *));
    if (worker->deque.buffer == NULL) {
        fprintf(stderr, "Failed to allocate deque buffer for worker %d\n", id);
        return;
    }

    // Initialize atomic indices
    atomic_init(&worker->deque.top, 0);
    atomic_init(&worker->deque.bottom, 0);

#ifdef DEBUG
    // Optional: warn if we rounded up
    if (actual_capacity != deque_capacity) {
        fprintf(stderr, "Worker %d: Rounded deque capacity %ld -> %ld\n", id, deque_capacity,
                actual_capacity);
    }
#endif
}

void worker_cleanup(Worker *worker) {
    if (worker == NULL) {
        return;
    }

    // Free the deque buffer
    // Note: We don't free individual tasks - they should be managed externally
    if (worker->deque.buffer != NULL) {
        free(worker->deque.buffer);
        worker->deque.buffer = NULL;
    }

    // Reset worker state
    worker->id = -1;
    worker->victim = -1;
    worker->stop_flag = 1;
    worker->deque.capacity = 0;
    worker->deque.mask = 0;
}

int32_t worker_select_victim(Worker *w, const int32_t num_workers) {
    if (w == NULL || num_workers <= 0) {
        return -1;
    }
    int32_t victim = -1;
    do {
        const uint64_t rand_val = xorshift64(&w->rng_state);
        victim = (int32_t)(rand_val % (uint64_t)num_workers);
    } while (victim == w->id);

    return victim;
}


void *worker_thread_func(void *arg) {
    Worker *worker = (Worker *)arg;
    ThreadPool *pool = worker->pool;

    pthread_barrier_wait(&pool->start_barrier);

    worker_main_loop(worker, pool);

    return NULL;
}

static bool deque_is_all_empty(ThreadPool *pool) {
    for (int i = 0; i < pool->num_workers; i++) {
        int64_t top = atomic_load(&pool->workers[i].deque.top);
        int64_t bottom = atomic_load(&pool->workers[i].deque.bottom);
        if (bottom > top)
            return false;
    }
    return true;
}

void worker_main_loop(struct Worker *worker, struct ThreadPool *pool) {
    current_worker = worker;
    current_pool = pool;

    int idle_count = 0;

    while (!atomic_load_explicit(&pool->shutdown, memory_order_acquire)) {

        Task *task = deque_pop_bottom(&worker->deque);

        if (!task) {
            // Try stealing
            for (int attempt = 0; attempt < NUM_STEAL_MAX_ATTEMPTS; attempt++) {
                int32_t victim_id = worker_select_victim(worker, pool->num_workers);
                if (victim_id < 0)
                    break;
                Worker *victim = &pool->workers[victim_id];
                task = deque_steal_top(&victim->deque);
                if (task)
                    break;
            }
        }

        if (task) {
            // Do work
            static _Atomic(int64_t) task_counter = 0;

            if (task->func)
                task->func(task);

            // decrement active_tasks
            int64_t prev = atomic_fetch_sub_explicit(
                &pool->active_tasks, 1, memory_order_acq_rel
                );

            // signal main thread IF we reached 0
            if (prev == 1) {
                pthread_mutex_lock(&pool->tasks_done_mutex);
                pthread_cond_broadcast(&pool->tasks_done_cond);
                pthread_mutex_unlock(&pool->tasks_done_mutex);
            }

            idle_count = 0;
            continue;
        }

        pthread_mutex_lock(&pool->work_mutex);
        while (!atomic_load_explicit(&pool->shutdown, memory_order_acquire) &&
               atomic_load_explicit(&pool->active_tasks, memory_order_acquire) == 0 &&
               deque_is_all_empty(pool)) {
            pthread_cond_wait(&pool->work_available, &pool->work_mutex);
        }

        pthread_mutex_unlock(&pool->work_mutex);

        worker_backoff(++idle_count);
    }
}


void worker_backoff(int idle_count) {
    if (idle_count < 10) {
        sched_yield();
    } else if (idle_count < 100) {
        struct timespec ts = {0, 1000}; // 1 μs
        nanosleep(&ts, NULL);
    } else {
        struct timespec ts = {0, 100000}; // 100 μs
        nanosleep(&ts, NULL);
    }
}

Worker *worker_current(void) {
    return current_worker;
}

ThreadPool *worker_current_pool(void) {
    return current_pool;
}