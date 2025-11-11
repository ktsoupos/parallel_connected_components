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
#    src/cc_sequential.c
)

# Link math library
target_link_libraries(cc_sequential m)

# TODO: Add parallel versions later
# - cc_openmp
# - cc_pthreads
# - cc_opencilk
