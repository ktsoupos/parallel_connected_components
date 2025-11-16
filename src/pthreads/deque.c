#include "deque.h"
#include <stdatomic.h>
#include <stddef.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"

bool deque_push_bottom(Deque* dq, Task* task)
{
    if (dq == NULL || task == NULL)
    {
        return false;
    }

    const int64_t bottom = atomic_load_explicit(&dq->bottom, memory_order_relaxed);
    const int64_t top = atomic_load_explicit(&dq->top, memory_order_acquire);

    if (bottom - top >= dq->capacity)
    {
        return false;
    }

    dq->buffer[bottom & dq->mask] = task;
    atomic_store_explicit(&dq->bottom, bottom + 1, memory_order_release);

    return true;
}

Task* deque_pop_bottom(Deque* dq)
{
    if (dq == NULL)
    {
        return NULL;
    }

    int64_t bottom = atomic_load_explicit(&dq->bottom, memory_order_relaxed);
    bottom--;
    atomic_store_explicit(&dq->bottom, bottom, memory_order_relaxed);

    atomic_thread_fence(memory_order_seq_cst);

    const int64_t top = atomic_load_explicit(&dq->top, memory_order_acquire);

    if (bottom < top)
    {
        // Empty
        atomic_store_explicit(&dq->bottom, top, memory_order_relaxed);
        return NULL;
    }

    Task* task = dq->buffer[bottom & dq->mask];

    if (bottom > top)
    {
        // More than one element, no race
        return task;
    }

    // Last element - race with stealers
    int64_t expected = top;
    if (!atomic_compare_exchange_strong_explicit(&dq->top, &expected, top + 1,
                                                  memory_order_seq_cst,
                                                  memory_order_relaxed))
    {
        task = NULL;
    }

    atomic_store_explicit(&dq->bottom, top + 1, memory_order_relaxed);
    return task;
}

Task* deque_steal_top(Deque* dq)
{
    if (dq == NULL)
    {
        return NULL;
    }

    int64_t top = atomic_load_explicit(&dq->top, memory_order_acquire);
    atomic_thread_fence(memory_order_seq_cst);
    const int64_t bottom = atomic_load_explicit(&dq->bottom, memory_order_acquire);

    if (top >= bottom)
    {
        return NULL;
    }

    Task* task = dq->buffer[top & dq->mask];

    // Try to claim it with CAS
    int64_t expected = top;
    if (!atomic_compare_exchange_strong_explicit(&dq->top, &expected, top + 1,
                                                  memory_order_seq_cst,
                                                  memory_order_relaxed))
    {
        return NULL;  // Lost the race
    }

    return task;  // Successfully stolen
}

#pragma GCC diagnostic pop