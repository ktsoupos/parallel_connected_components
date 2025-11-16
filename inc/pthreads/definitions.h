#pragma once
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct ThreadPool ThreadPool; // Forward declaration

typedef struct Task {
    void (*func)(struct Task *t); // function to execute this task
    int64_t start_vertex;         // first vertex in this chunk
    int64_t end_vertex;           // one past the last vertex
    void *graph;                  // pointer to graph data
    char pad[32];                 // padding: 64 - (8+8+8+8) = 32 bytes
} Task __attribute__((aligned(64)));


typedef struct Deque {
    Task **buffer;            // array of pointers to tasks
    int64_t capacity;         // maximum number of tasks it can hold
    int64_t mask;             // capacity - 1
    _Atomic(int64_t) top;     // index for thieves
    _Atomic(int64_t) bottom;  // index for owner only
    char pad[24];             // padding: 64 - (8+8+8+8+8) = 24 bytes
} Deque __attribute__((aligned(64)));


typedef struct Worker {
    Deque deque;              // 64 bytes (embedded aligned struct)
    uint64_t rng_state;       // 8 bytes - XORShift RNG state
    ThreadPool *pool;         // 8 bytes - pointer to parent thread pool
    int32_t id;               // 4 bytes - worker ID
    int32_t victim;           // 4 bytes - last selected victim
    _Atomic(int8_t) stop_flag; // 1 byte - stop signal
    char pad[39];             // padding: 128 - (64+8+8+4+4+1) = 39 bytes
} Worker __attribute__((aligned(64)));

struct ThreadPool {
    Worker *workers;                   // 8 bytes - array of workers
    pthread_t *threads;                // 8 bytes - thread handles
    _Atomic(int64_t) active_tasks;     // 8 bytes - tasks in system
    pthread_barrier_t start_barrier;   // 32 bytes (typical x86_64) - sync start
    pthread_mutex_t tasks_done_mutex;  // 40 bytes (typical x86_64) - for condition variable
    pthread_cond_t tasks_done_cond;    // 48 bytes (typical x86_64) - signal task completion
    int32_t num_workers;               // 4 bytes - number of workers
    _Atomic(int8_t) shutdown;          // 1 byte - shutdown signal (0=running, 1=shutdown)
    char pad[35];                      // padding: 192 - (8+8+8+32+40+48+4+1) = 43 bytes
} __attribute__((aligned(64)));

