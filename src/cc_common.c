#include "cc_common.h"
#include <stdlib.h>
#include <stdio.h>

int32_t count_unique_labels(const int32_t* labels, const int32_t num_vertices) {
    if (labels == NULL || num_vertices <= 0) {
        return -1;
    }

    bool* seen = calloc((size_t)num_vertices, sizeof(bool));
    if (seen == NULL) {
        fprintf(stderr, "Error: Failed to allocate seen array\n");
        return -1;
    }

    int32_t count = 0;
    for (int32_t v = 0; v < num_vertices; v++) {
        const int32_t label = labels[v];

        /* Bounds check to prevent heap corruption */
        if (label < 0 || label >= num_vertices) {
            fprintf(stderr, "Error: Invalid label %d for vertex %d (range: 0-%d)\n",
                    label, v, num_vertices - 1);
            free(seen);
            return -1;
        }

        if (!seen[label]) {
            seen[label] = true;
            count++;
        }
    }

    free(seen);
    return count;
}

void print_component_stats(const int32_t* labels, const int32_t num_vertices) {
    if (labels == NULL || num_vertices <= 0) {
        fprintf(stderr, "Error: Invalid parameters\n");
        return;
    }

    /* Count size of each component */
    int32_t* component_sizes = calloc((size_t)num_vertices, sizeof(int32_t));
    if (component_sizes == NULL) {
        fprintf(stderr, "Error: Failed to allocate component sizes array\n");
        return;
    }

    for (int32_t v = 0; v < num_vertices; v++) {
        const int32_t label = labels[v];

        /* Bounds check to prevent heap corruption */
        if (label < 0 || label >= num_vertices) {
            fprintf(stderr, "Error: Invalid label %d for vertex %d\n", label, v);
            free(component_sizes);
            return;
        }

        component_sizes[label]++;
    }

    /* Find statistics */
    int32_t min_size = num_vertices;
    int32_t max_size = 0;
    int32_t num_components = 0;
    double total_size = 0.0;

    for (int32_t v = 0; v < num_vertices; v++) {
        if (component_sizes[v] > 0) {
            num_components++;
            total_size += (double)component_sizes[v];

            if (component_sizes[v] < min_size) {
                min_size = component_sizes[v];
            }
            if (component_sizes[v] > max_size) {
                max_size = component_sizes[v];
            }
        }
    }

    const double avg_size = total_size / (double)num_components;

    printf("  Component size statistics:\n");
    printf("    Min: %d vertices\n", min_size);
    printf("    Max: %d vertices\n", max_size);
    printf("    Average: %.2f vertices\n", avg_size);

    free(component_sizes);
}
