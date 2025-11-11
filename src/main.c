#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include "graph.h"
#include "mtx_reader.h"
#include "cc_sequential.h"

int main(const int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph.mtx> [report_interval]\n", argv[0]);
        fprintf(stderr, "  graph.mtx: Matrix Market format graph file\n");
        fprintf(stderr, "  report_interval: Optional progress report interval (0 = silent)\n");
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];
    int32_t report_interval = 0;

    if (argc >= 3) {
        char* endptr;
        errno = 0;
        const long val = strtol(argv[2], &endptr, 10);

        if (errno != 0 || endptr == argv[2] || *endptr != '\0' || val < 0 || val > INT32_MAX) {
            fprintf(stderr, "Error: Invalid report_interval value '%s'\n", argv[2]);
            return EXIT_FAILURE;
        }

        report_interval = (int32_t)val;
    }

    printf("=== Connected Components - Sequential Version ===\n\n");

    /* Read graph from MTX file */
    Graph* g = read_mtx_file_verbose(filename, report_interval);
    if (g == NULL) {
        fprintf(stderr, "Error: Failed to read graph from '%s'\n", filename);
        return EXIT_FAILURE;
    }

    printf("\n");
    graph_print_stats(g);

    /* Run connected components algorithm */
    printf("\nRunning connected components algorithm...\n");

    const clock_t start = clock();
    CCResult* result = label_propagation_min(g);
    const clock_t end = clock();

    if (result == NULL) {
        fprintf(stderr, "Error: Connected components algorithm failed\n");
        graph_destroy(g);
        return EXIT_FAILURE;
    }

    const double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Algorithm completed in %.3f seconds\n", elapsed);

    /* Print results */
    cc_result_print_stats(result, g);

    /* Cleanup */
    cc_result_destroy(result);
    graph_destroy(g);

    return EXIT_SUCCESS;
}
