#pragma once
#include "definitions.h"

void worker_init(Worker *worker, int32_t id, int64_t deque_capacity);

void *worker_thread_func(void *arg);

void worker_main_loop(Worker *worker, ThreadPool *pool);

int32_t worker_select_victim(Worker *w, int32_t num_workers);

void worker_backoff(int idle_count);

void worker_cleanup(Worker *worker);

/**
 * Get current worker from thread-local storage
 * Only valid when called from within a worker thread
 */
Worker *worker_current(void);

/**
 * Get current thread pool from thread-local storage
 * Only valid when called from within a worker thread
 */
ThreadPool *worker_current_pool(void);
