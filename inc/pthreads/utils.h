#pragma once

#include <stdbool.h>
#include <stdint.h>

/**
 * Check if a number is a power of 2
 */
static inline bool is_power_of_two(int64_t x) {
    return (x > 0) && ((x & (x - 1)) == 0);
}

/**
 * Round up to next power of 2
 * If x is already a power of 2, returns x unchanged
 * Returns 1 for x <= 1
 */
static inline int64_t next_power_of_two(int64_t x) {
    if (x <= 1) {
        return 1;
    }
    if (is_power_of_two(x)) {
        return x;
    }

#if defined(__GNUC__) || defined(__clang__)
    // Fast version using compiler builtin
    return 1LL << (64 - __builtin_clzll((uint64_t)x - 1));
#else
    // Portable fallback using bit manipulation
    uint64_t v = (uint64_t)(x - 1);
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return (int64_t)(v + 1);
#endif
}

/**
 * 32-bit version for smaller values
 */
static inline int32_t next_power_of_two_32(int32_t x) {
    if (x <= 1) {
        return 1;
    }
    if ((x > 0) && ((x & (x - 1)) == 0)) {
        return x;
    }

#if defined(__GNUC__) || defined(__clang__)
    return 1 << (32 - __builtin_clz((uint32_t)x - 1));
#else
    uint32_t v = (uint32_t)(x - 1);
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return (int32_t)(v + 1);
#endif
}
