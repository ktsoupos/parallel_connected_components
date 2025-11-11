#include "cc_sequential.h"
#include "cc_common.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

CCResult* label_propagation_min(const Graph* g) {
    /* Check arguments*/
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);
    if (num_vertices <= 0) {
        fprintf(stderr, "Error: Invalid number of vertices\n");
        return NULL;
    }

    /* Allocate result structure */
    CCResult* result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        return NULL;
    }

    /* Allocate labels array */
    result->labels = malloc(sizeof(int32_t) * (size_t)num_vertices);
    if (result->labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate labels array\n");
        free(result);
        return NULL;
    }

    /* Allocate queue arrays */
    int32_t* queue = malloc(sizeof(int32_t) * (size_t)num_vertices);
    int32_t* next_queue = malloc(sizeof(int32_t) * (size_t)num_vertices);
    bool* in_queue = calloc((size_t)num_vertices, sizeof(bool));

    if ((queue == NULL) || (next_queue == NULL) || (in_queue == NULL)) {
        fprintf(stderr, "Error: Failed to allocate queue arrays\n");
        free(result->labels);
        free(result);
        free(queue);      /* Safe: free(NULL) is a no-op */
        free(next_queue); /* Safe: free(NULL) is a no-op */
        free(in_queue);   /* Safe: free(NULL) is a no-op */
        return NULL;
    }

    /* Keep original pointers for cleanup (since we swap pointers in the loop) */
    int32_t* queue_orig = queue;
    int32_t* next_queue_orig = next_queue;

    /* Initialize: each vertex gets its own label, all vertices in queue */
    for (int32_t i = 0; i < num_vertices; i++) {
        result->labels[i] = i;
        queue[i] = i;
        in_queue[i] = true;  /* Mark all vertices as in queue initially */
    }
    int32_t queue_size = num_vertices;

    /* Label propagation with queue optimization */
    result->num_iterations = 0;

    while (queue_size > 0) {
        result->num_iterations++;
        int32_t next_size = 0;

        /* Clear in_queue flags for next iteration */
        for (int32_t i = 0; i < queue_size; i++) {
            in_queue[queue[i]] = false;
        }

        /* Process only vertices in current queue */
        for (int32_t i = 0; i < queue_size; i++) {
            const int32_t v = queue[i];

            /* Bounds check vertex from queue */
            if ((v < 0) || (v >= num_vertices)) {
                fprintf(stderr, "Error: Invalid vertex %d in queue (range: 0-%d)\n",
                        v, num_vertices - 1);
                free(queue_orig);
                free(next_queue_orig);
                free(in_queue);
                free(result->labels);
                free(result);
                return NULL;
            }

            bool has_changed = false;
            int32_t num_neighbors;
            const int32_t* neighbors = graph_get_neighbors(g, v, &num_neighbors);

            if (neighbors == NULL) {
                continue;
            }

            /* Find minimum label among neighbors */
            for (int32_t j = 0; j < num_neighbors; j++) {
                const int32_t u = neighbors[j];

                /* Bounds check to prevent heap corruption */
                if ((u < 0) || (u >= num_vertices)) {
                    fprintf(stderr, "Error: Invalid neighbor index %d for vertex %d (range: 0-%d)\n",
                            u, v, num_vertices - 1);
                    free(queue_orig);
                    free(next_queue_orig);
                    free(in_queue);
                    free(result->labels);
                    free(result);
                    return NULL;
                }

                if (result->labels[v] > result->labels[u]) {
                    result->labels[v] = result->labels[u];
                    has_changed = true;
                }
            }

            /* If label changed, add neighbors to next queue */
            if (has_changed) {
                for (int32_t j = 0; j < num_neighbors; j++) {
                    const int32_t u = neighbors[j];

                    /* u already validated in previous loop */
                    if (!in_queue[u]) {  /* Avoid duplicates */
                        /* Safety check: prevent queue overflow */
                        if (next_size >= num_vertices) {
                            fprintf(stderr, "Error: Queue overflow at iteration %d\n",
                                    result->num_iterations);
                            free(queue_orig);
                            free(next_queue_orig);
                            free(in_queue);
                            free(result->labels);
                            free(result);
                            return NULL;
                        }
                        next_queue[next_size++] = u;
                        in_queue[u] = true;
                    }
                }
            }
        }

        /* Swap queues */
        int32_t* temp = queue;
        queue = next_queue;
        next_queue = temp;
        queue_size = next_size;
    }

    /* Cleanup queue arrays - use original pointers */
    free(queue_orig);
    free(next_queue_orig);
    free(in_queue);

    /* Count connected components */
    result->num_components = count_unique_labels(result->labels, num_vertices);
    if (result->num_components < 0) {
        fprintf(stderr, "Error: Failed to count components\n");
        free(result->labels);
        free(result);
        return NULL;
    }

    return result;
}

void cc_result_destroy(CCResult* result) {
    if (result == NULL) {
        return;
    }

    free(result->labels);
    free(result);
}

void cc_result_print_stats(const CCResult* result, const Graph* g) {
    if (result == NULL) {
        fprintf(stderr, "Error: NULL result pointer\n");
        return;
    }

    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);

    printf("\nConnected Components Results:\n");
    printf("  Number of components: %d\n", result->num_components);
    printf("  Iterations to converge: %d\n", result->num_iterations);

    /* Print component size statistics */
    print_component_stats(result->labels, num_vertices);
}
