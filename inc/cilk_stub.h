/**
 * Cilk Compatibility Header for CLion IDE
 *
 * This header helps CLion's parser understand Cilk keywords.
 * It does NOT affect the actual build - OpenCilk compiler ignores these definitions.
 *
 * Usage: Include this BEFORE <cilk/cilk.h> in OpenCilk source files
 */

#pragma once

// Detect if we're in IDE parsing mode (not actual compilation)
// CLion defines __JETBRAINS_IDE__ when parsing code
// We also check if we're NOT using the OpenCilk compiler
#if defined(__JETBRAINS_IDE__) || defined(__CLION_IDE__) || !defined(__cilk)

// Only define these if cilk/cilk.h hasn't been included yet
// This prevents conflicts with actual Cilk headers

#ifndef __CILKRTS_CILK_H_INCLUDED

// Define Cilk keywords as no-ops for IDE parser
// This removes red squiggles while keeping actual build working

#define cilk_for for
#define cilk_spawn
#define cilk_sync

// Also handle the underscore versions (used internally by cilk/cilk.h)
#define _Cilk_for for
#define _Cilk_spawn
#define _Cilk_sync

// Prevent the real cilk.h from redefining these when IDE is parsing
#define __CILKRTS_CILK_H_INCLUDED

#endif // __CILKRTS_CILK_H_INCLUDED

#endif // IDE detection
