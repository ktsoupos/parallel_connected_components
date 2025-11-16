#pragma once
#include "definitions.h"

/**
 * Create a thread pool with num_workers threads
 * Each worker gets a deque with deque_capacity
 */
ThreadPool *threadpool_create(int32_t num_workers, int64_t deque_capacity);

/**
 * Start all worker threads
 */
void threadpool_start(ThreadPool *pool);

/**
 * Wait for all tasks to complete
 */
void threadpool_wait(ThreadPool *pool);

/**
 * Shutdown the thread pool and join all threads
 */
void threadpool_shutdown(ThreadPool *pool);

/**
 * Destroy the thread pool and free resources
 */
void threadpool_destroy(ThreadPool *pool);