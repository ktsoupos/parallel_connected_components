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
include_directories(
        inc
        inc/pthreads
)

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

# OpenCilk parallel version
# To build with OpenCilk, configure cmake with: CC=/opt/opencilk/bin/clang cmake -B cmake-build-opencilk
if(EXISTS "/opt/opencilk/bin/clang")
    message(STATUS "OpenCilk compiler found at /opt/opencilk/bin/clang")

    # Check if we're using the OpenCilk compiler
    if(CMAKE_C_COMPILER MATCHES "opencilk")
        message(STATUS "Building with OpenCilk compiler")

        # Define common source files for OpenCilk targets
        set(OPENCILK_SOURCES
            src/main.c
            src/graph.c
            src/mtx_reader.c
            src/cc_common.c
            src/cc_sequential.c
            src/cc_opencilk.c
            benchmarks/benchmark_sequential.c
            benchmarks/benchmark_opencilk.c
        )

        # 1. Standard OpenCilk executable (optimized)
        add_executable(cc_opencilk ${OPENCILK_SOURCES})
        target_compile_options(cc_opencilk PRIVATE -fopencilk -O3)
        target_link_options(cc_opencilk PRIVATE -fopencilk)
        target_link_libraries(cc_opencilk m)

        # 2. Cilksan - Race detector
        # Usage: ./cmake-build-opencilk/cc_opencilk_cilksan <args>
        # Runs serially and detects all determinacy races
        add_executable(cc_opencilk_cilksan ${OPENCILK_SOURCES})
        target_compile_options(cc_opencilk_cilksan PRIVATE
            -fopencilk
            -fsanitize=cilk
            -Og
            -g
        )
        target_link_options(cc_opencilk_cilksan PRIVATE
            -fopencilk
            -fsanitize=cilk
        )
        target_link_libraries(cc_opencilk_cilksan m)

        # 3. Cilkscale - Work-span analyzer
        # Usage: ./cmake-build-opencilk/cc_opencilk_cilkscale <args>
        # Outputs work, span, and parallelism measurements in CSV format
        # Set CILKSCALE_OUT=output.csv to write to file
        add_executable(cc_opencilk_cilkscale ${OPENCILK_SOURCES})
        target_compile_options(cc_opencilk_cilkscale PRIVATE
            -fopencilk
            -fcilktool=cilkscale
            -O3
        )
        target_link_options(cc_opencilk_cilkscale PRIVATE
            -fopencilk
            -fcilktool=cilkscale
        )
        target_link_libraries(cc_opencilk_cilkscale m)

        # 4. Cilkscale Benchmark - For scalability benchmarking
        # Usage with Python script:
        # python3 /opt/opencilk/share/Cilkscale_vis/cilkscale.py \
        #   -c ./cmake-build-opencilk/cc_opencilk_cilkscale \
        #   -b ./cmake-build-opencilk/cc_opencilk_cilkscale_bench \
        #   --args <your_args>
        add_executable(cc_opencilk_cilkscale_bench ${OPENCILK_SOURCES})
        target_compile_options(cc_opencilk_cilkscale_bench PRIVATE
            -fopencilk
            -fcilktool=cilkscale-benchmark
            -O3
        )
        target_link_options(cc_opencilk_cilkscale_bench PRIVATE
            -fopencilk
            -fcilktool=cilkscale-benchmark
        )
        target_link_libraries(cc_opencilk_cilkscale_bench m)

        message(STATUS "OpenCilk targets configured:")
        message(STATUS "  - cc_opencilk: Standard optimized build")
        message(STATUS "  - cc_opencilk_cilksan: Race detector")
        message(STATUS "  - cc_opencilk_cilkscale: Work-span analyzer")
        message(STATUS "  - cc_opencilk_cilkscale_bench: Scalability benchmarking")
    else()
        message(STATUS "To build OpenCilk targets, reconfigure with: CC=/opt/opencilk/bin/clang cmake -B cmake-build-opencilk")
    endif()
else()
    message(WARNING "OpenCilk compiler not found at /opt/opencilk/bin/clang")
endif()

# Pthreads parallel version
find_package(Threads REQUIRED)
if(Threads_FOUND)
    message(STATUS "Pthreads found")

    # Pthreads parallel version
    add_executable(cc_pthreads
        src/main.c
        src/graph.c
        src/mtx_reader.c
        src/cc_common.c
        src/cc_sequential.c
#        src/cc_pthreads.c
        src/pthreads/deque.c
        benchmarks/benchmark_sequential.c
#        benchmarks/benchmark_pthreads.c
    )

    # Link Pthreads and math library
    target_link_libraries(cc_pthreads Threads::Threads m)

    # Add pthread-specific compiler flags
    target_compile_options(cc_pthreads PRIVATE -pthread)
else()
    message(WARNING "Pthreads not found - cc_pthreads target will not be built")
endif()
