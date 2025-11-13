#include "cc_common.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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

int32_t sample_frequent_element(const int32_t *comp, int32_t num_vertices, int32_t num_samples) {
    if (comp == NULL || num_vertices <= 0 || num_samples <= 0) {
        fprintf(stderr, "Error: Invalid parameters for sample_frequent_element\n");
        return -1;
    }

    /* Allocate counter array for tracking sample counts */
    int32_t *sample_counts = calloc((size_t)num_vertices, sizeof(int32_t));
    if (sample_counts == NULL) {
        fprintf(stderr, "Error: Failed to allocate sample_counts array\n");
        return -1;
    }

    /* Seed random number generator (use time + address for better randomness) */
    static bool seeded = false;
    if (!seeded) {
        srand((unsigned int)(time(NULL) ^ (time_t)(uintptr_t)comp));
        seeded = true;
    }

    /* Sample random elements from comp array */
    for (int32_t i = 0; i < num_samples; i++) {
        /* Generate random index in range [0, num_vertices) */
        int32_t idx = rand() % num_vertices;
        int32_t component_id = comp[idx];

        /* Bounds check to prevent heap corruption */
        if (component_id < 0 || component_id >= num_vertices) {
            fprintf(stderr, "Error: Invalid component ID %d at index %d\n", component_id, idx);
            free(sample_counts);
            return -1;
        }

        sample_counts[component_id]++;
    }

    /* Find the most frequent element */
    int32_t most_frequent_id = 0;
    int32_t max_count = sample_counts[0];

    for (int32_t i = 1; i < num_vertices; i++) {
        if (sample_counts[i] > max_count) {
            max_count = sample_counts[i];
            most_frequent_id = i;
        }
    }

    /* Calculate and print percentage */
    float percentage = (float)max_count / (float)num_samples * 100.0f;
    printf("Skipping largest intermediate component (ID: %d, approx. %.1f%% of the graph)\n",
           most_frequent_id, percentage);

    free(sample_counts);
    return most_frequent_id;
}
