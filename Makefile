# Makefile for Parallel Connected Components
# Provides simple interface for building all targets

.PHONY: all clean sequential openmp pthreads opencilk test help

# Default target
all: sequential openmp pthreads
	@echo "==================================="
	@echo "Build complete!"
	@echo "==================================="
	@echo "Executables:"
	@echo "  build/cc_sequential - Sequential implementation"
	@echo "  build/cc_openmp     - OpenMP parallel implementation"
	@echo "  build/cc_pthreads   - PThreads work-stealing implementation"
	@echo ""
	@echo "To build OpenCilk version:"
	@echo "  make opencilk"
	@echo ""
	@echo "To run tests:"
	@echo "  make test"

# Build directory setup
build:
	@mkdir -p build

# Sequential + OpenMP + PThreads (standard build)
sequential openmp pthreads: build
	@echo "Building standard targets (Sequential, OpenMP, PThreads)..."
	cmake -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j$(shell nproc)

# OpenCilk build (requires OpenCilk compiler)
opencilk:
	@echo "Building OpenCilk targets..."
	@if [ ! -f /opt/opencilk/bin/clang ]; then \
		echo "ERROR: OpenCilk compiler not found at /opt/opencilk/bin/clang"; \
		echo "Please install OpenCilk first."; \
		exit 1; \
	fi
	CC=/opt/opencilk/bin/clang cmake -B cmake-build-opencilk -DCMAKE_BUILD_TYPE=Release
	cmake --build cmake-build-opencilk -j$(shell nproc)
	@echo ""
	@echo "OpenCilk executables built in cmake-build-opencilk/:"
	@echo "  cc_opencilk               - Standard optimized build"
	@echo "  cc_opencilk_cilksan       - Race detector"
	@echo "  cc_opencilk_cilkscale     - Work-span analyzer"
	@echo "  cc_opencilk_cilkscale_bench - Scalability benchmarking"

# Run basic correctness test
test: all
	@echo "==================================="
	@echo "Running correctness tests..."
	@echo "==================================="
	@echo ""
	@echo "Test 1: Small graph (6 vertices, 2 components)"
	@echo "-----------------------------------"
	@./build/cc_sequential data/test_small.mtx 0 | grep -E "components|vertices"
	@echo ""
	@echo "Test 2: Sequential vs OpenMP consistency"
	@echo "-----------------------------------"
	@./build/cc_openmp data/test_small.mtx 0 4 | grep -E "components|vertices"
	@echo ""
	@echo "Test 3: PThreads consistency"
	@echo "-----------------------------------"
	@./build/cc_pthreads data/test_small.mtx 0 4 | grep -E "components|vertices"
	@echo ""
	@echo "All tests should show: 6 vertices, 2 components"

# Run performance benchmarks
benchmark: all
	@echo "==================================="
	@echo "Performance Benchmarks"
	@echo "==================================="
	@echo ""
	@if [ -f data/ca-CondMat.mtx ]; then \
		echo "Running on ca-CondMat.mtx..."; \
		./scripts/run_benchmarks.sh data/ca-CondMat.mtx; \
	elif [ -f data/test_small.mtx ]; then \
		echo "Running on test_small.mtx (demo only)..."; \
		./scripts/run_benchmarks.sh data/test_small.mtx; \
	else \
		echo "No test data found in data/ directory"; \
		echo "Please add .mtx files to data/ directory"; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build cmake-build-* CMakeFiles CMakeCache.txt
	@echo "Clean complete!"

# Help message
help:
	@echo "Parallel Connected Components - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make                 - Build all standard targets (sequential, OpenMP, PThreads)"
	@echo "  make all             - Same as default"
	@echo "  make sequential      - Build only sequential version"
	@echo "  make openmp          - Build only OpenMP version"
	@echo "  make pthreads        - Build only PThreads version"
	@echo "  make opencilk        - Build OpenCilk version (requires OpenCilk compiler)"
	@echo "  make test            - Run correctness tests"
	@echo "  make benchmark       - Run performance benchmarks"
	@echo "  make clean           - Remove all build artifacts"
	@echo "  make help            - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - CMake 3.19.2+"
	@echo "  - GCC 11.4+ (with OpenMP support)"
	@echo "  - OpenCilk (optional, for opencilk target)"
	@echo ""
	@echo "Examples:"
	@echo "  make && make test                    # Build and test"
	@echo "  make benchmark                        # Run performance tests"
	@echo "  ./build/cc_sequential data/graph.mtx # Run sequential on custom graph"
