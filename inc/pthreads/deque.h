#pragma once
#include <definitions.h>
#include <stdbool.h>

bool deque_push_bottom(Deque* dq, Task* task);

Task* deque_pop_bottom(Deque* dq);

Task* deque_steal_top(Deque* dq);