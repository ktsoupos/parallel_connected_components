#pragma once
#include <stdint.h>


typedef struct Task
{
    void (*func)(struct Task* t); // function to execute this task
    int64_t start_vertex; // first vertex in this chunk
    int64_t end_vertex; // one past the last vertex
    void* graph; // pointer to graph data
    char pad[64]; // optional padding to avoid false sharing
} Task __attribute__((aligned(64)));


typedef struct Deque
{
    Task** buffer; // array of pointers to tasks
    int64_t capacity; // maximum number of tasks it can hold
    int64_t mask; // capacity - 1
    _Atomic(int64_t) top; // index for thieves
    _Atomic(int64_t) bottom; // index for owner only
    char pad[64]; // ensures top/bottom of one deque don't share cache line with another
} Deque __attribute__((aligned(64)));
