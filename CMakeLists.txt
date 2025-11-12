cmake_minimum_required(VERSION 3.19.2)
project(parallel_connected_components C)

set(CMAKE_C_STANDARD 11)

# Compiler flags
add_compile_options(
    -Wall
    -Wextra
    -pedantic
    -Wconversion           # Warn on implicit conversions
    -Wsign-conversion      # Warn on sign conversions
    -Wmissing-prototypes   # Warn on missing prototypes
    -Wshadow               # Warn on variable shadowing
    -Wcast-qual            # Warn on cast removing qualifiers
    -Wcast-align           # Warn on alignment changes
    -Wwrite-strings        # Warn on string literal assignments
    -Wundef                # Warn on undefined macros
)
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Include directories
include_directories(inc)

# Sequential version
add_executable(cc_sequential
    src/main.c
    src/graph.c
    src/mtx_reader.c
    src/cc_common.c
    src/cc_sequential.c
    benchmarks/benchmark_sequential.c
)

# Link math library
target_link_libraries(cc_sequential m)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_C_VERSION}")

    # OpenMP parallel version
    add_executable(cc_openmp
        src/main.c
        src/graph.c
        src/mtx_reader.c
        src/cc_common.c
        src/cc_sequential.c
        src/cc_openmp.c
        benchmarks/benchmark_sequential.c
        benchmarks/benchmark_openmp.c
    )

    # Link OpenMP and math library
    target_link_libraries(cc_openmp OpenMP::OpenMP_C m)

    # Optional: Add OpenMP-specific flags for better performance
    target_compile_options(cc_openmp PRIVATE -fopenmp)
else()
    message(WARNING "OpenMP not found - cc_openmp target will not be built")
endif()

# TODO: Add other parallel versions later
# - cc_pthreads
# - cc_opencilk
