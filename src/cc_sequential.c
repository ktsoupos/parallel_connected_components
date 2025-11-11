#include "cc_sequential.h"
#include "cc_common.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

CCResult *label_propagation_min(const Graph *restrict g) {
    /* Check arguments */
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
    CCResult *restrict result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        return NULL;
    }

    /* Allocate labels array */
    result->labels = malloc(sizeof(int32_t) * (size_t) num_vertices);
    if (result->labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate labels array\n");
        free(result);
        return NULL;
    }

    /* Use restrict pointer for better optimization */
    int32_t *restrict labels = result->labels;

    /* Allocate temporary arrays in single block for better performance */
    const size_t queue_size_bytes = sizeof(int32_t) * (size_t) num_vertices;
    const size_t bool_size_bytes = sizeof(bool) * (size_t) num_vertices;

    /* Ensure proper alignment: round up bool_size to multiple of int32_t alignment */
    const size_t bool_size_aligned = ((bool_size_bytes + sizeof(int32_t) - 1) / sizeof(int32_t)) * sizeof(int32_t);
    const size_t total_bytes = (2 * queue_size_bytes) + bool_size_aligned;

    void *temp_memory = malloc(total_bytes);
    if (temp_memory == NULL) {
        fprintf(stderr, "Error: Failed to allocate temporary arrays\n");
        free(result->labels);
        free(result);
        return NULL;
    }

    /* Partition the single block into three arrays using pointer arithmetic
     * Using pointer arithmetic avoids alignment warnings and is cleaner */
    int32_t *queue = (int32_t *) temp_memory;
    int32_t *next_queue = queue + num_vertices; /* Advance by num_vertices int32_t elements */
    bool *in_queue = (bool *) (next_queue + num_vertices); /* Advance by another num_vertices */

    /* Keep original pointer for cleanup (since we swap queue pointers in the loop) */
    void *temp_memory_orig = temp_memory;

    /* Initialize: each vertex gets its own label, all vertices in queue */
    for (int32_t i = 0; i < num_vertices; i++) {
        labels[i] = i;
        queue[i] = i;
        in_queue[i] = false; /* in_queue tracks next_queue, initially empty */
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

#ifdef DEBUG
            /* Bounds check vertex from queue (debug only) */
            if ((v < 0) || (v >= num_vertices)) {
                fprintf(stderr, "Error: Invalid vertex %d in queue (range: 0-%d)\n",
                        v, num_vertices - 1);
                free(temp_memory_orig);
                free(result->labels);
                free(result);
                return NULL;
            }
#endif

            int32_t num_neighbors;
            const int32_t *restrict neighbors = graph_get_neighbors(g, v, &num_neighbors);

            if (neighbors == NULL) {
                continue;
            }

            /* Find minimum label among neighbors in single pass */
            int32_t min_label = labels[v];
            for (int32_t j = 0; j < num_neighbors; j++) {
                const int32_t u = neighbors[j];

#ifdef DEBUG
                /* Bounds check to prevent heap corruption (debug only) */
                if ((u < 0) || (u >= num_vertices)) {
                    fprintf(stderr, "Error: Invalid neighbor index %d for vertex %d (range: 0-%d)\n",
                            u, v, num_vertices - 1);
                    free(temp_memory_orig);
                    free(result->labels);
                    free(result);
                    return NULL;
                }
#endif

                if (labels[u] < min_label) {
                    min_label = labels[u];
                }
            }

            /* Only update and propagate if label actually changed */
            if (min_label < labels[v]) {
                labels[v] = min_label;

                /* Add neighbors to next queue for propagation */
                for (int32_t j = 0; j < num_neighbors; j++) {
                    const int32_t u = neighbors[j];

                    if (!in_queue[u]) {
#ifdef DEBUG
                        /* Queue overflow check (debug only) */
                        if (next_size >= num_vertices) {
                            fprintf(stderr, "Error: Queue overflow at iteration %d\n",
                                    result->num_iterations);
                            free(temp_memory_orig);
                            free(result->labels);
                            free(result);
                            return NULL;
                        }
#endif
                        next_queue[next_size++] = u;
                        in_queue[u] = true;
                    }
                }
            }
        }

        /* Swap queues */
        int32_t *temp = queue;
        queue = next_queue;
        next_queue = temp;
        queue_size = next_size;
    }

    /* Cleanup temporary arrays - single free */
    free(temp_memory_orig);

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

CCResult *label_propagation_min_simple(const Graph *restrict g) {
    /* Check arguments */
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
    CCResult *restrict result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        return NULL;
    }

    /* Allocate labels array */
    result->labels = malloc(sizeof(int32_t) * (size_t) num_vertices);
    if (result->labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate labels array\n");
        free(result);
        return NULL;
    }

    int32_t *restrict labels = result->labels;

    /* Initialize: each vertex gets its own label */
    for (int32_t i = 0; i < num_vertices; i++) {
        labels[i] = i;
    }

    /* Simple label propagation - process all vertices each iteration */
    result->num_iterations = 0;
    bool changed = true;

    while (changed) {
        result->num_iterations++;
        changed = false;

        /* Process all vertices */
        for (int32_t v = 0; v < num_vertices; v++) {
            int32_t num_neighbors;
            const int32_t *restrict neighbors = graph_get_neighbors(g, v, &num_neighbors);

            if (neighbors == NULL) {
                continue;
            }

            /* Find minimum label among neighbors */
            int32_t min_label = labels[v];
            for (int32_t j = 0; j < num_neighbors; j++) {
                const int32_t u = neighbors[j];

                if (labels[u] < min_label) {
                    min_label = labels[u];
                }
            }

            /* Update if found smaller label */
            if (min_label < labels[v]) {
                labels[v] = min_label;
                changed = true;
            }
        }
    }

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

/* Union-Find helper: find with iterative path compression */
static int32_t uf_find(int32_t *restrict parent, int32_t x) {
    /* Find root */
    int32_t root = x;
    while (parent[root] != root) {
        root = parent[root];
    }

    /* Path compression: point all nodes along path directly to root */
    while (parent[x] != root) {
        const int32_t next = parent[x];
        parent[x] = root;
        x = next;
    }

    return root;
}

/* Union-Find helper: union by minimum ID
 * Always attach higher-ID root to lower-ID root
 * This directly produces minimum labels without normalization! */
static void uf_union(int32_t *restrict parent, const int32_t x, const int32_t y) {
    const int32_t root_x = uf_find(parent, x);
    const int32_t root_y = uf_find(parent, y);

    if (root_x == root_y) {
        return;  /* Already in same set */
    }

    /* Union by minimum: attach higher ID to lower ID */
    if (root_x < root_y) {
        parent[root_y] = root_x;
    } else {
        parent[root_x] = root_y;
    }
}

CCResult *union_find_cc(const Graph *restrict g) {
    /* Check arguments */
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
    CCResult *restrict result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        return NULL;
    }

    /* Allocate parent array */
    int32_t *parent = malloc(sizeof(int32_t) * (size_t) num_vertices);

    if (parent == NULL) {
        fprintf(stderr, "Error: Failed to allocate parent array\n");
        free(result);
        return NULL;
    }

    for (int32_t i = 0; i < num_vertices; i++) {
        parent[i] = i;
    }

    /* Process all edges: union endpoints */
    for (int32_t v = 0; v < num_vertices; v++) {
        int32_t num_neighbors = 0;
        const int32_t *restrict neighbors = graph_get_neighbors(g, v, &num_neighbors);

        if (neighbors == NULL) {
            continue;
        }

        for (int32_t j = 0; j < num_neighbors; j++) {
            const int32_t u = neighbors[j];
            uf_union(parent, v, u);
        }
    }

    /* Allocate labels array */
    result->labels = malloc(sizeof(int32_t) * (size_t) num_vertices);
    if (result->labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate labels array\n");
        free(parent);
        free(result);
        return NULL;
    }

    /* Final path compression: assign minimum labels
     * Union-by-minimum ensures roots are already minimum IDs */
    for (int32_t i = 0; i < num_vertices; i++) {
        result->labels[i] = uf_find(parent, i);
    }

    /* Union-Find processes edges once, no iterations */
    result->num_iterations = 1;

    /* Count connected components */
    result->num_components = count_unique_labels(result->labels, num_vertices);
    if (result->num_components < 0) {
        fprintf(stderr, "Error: Failed to count components\n");
        free(parent);
        free(result->labels);
        free(result);
        return NULL;
    }

    /* Cleanup parent array */
    free(parent);

    return result;
}

/* Non-restrict versions for benchmarking */
static int32_t uf_find_no_restrict(int32_t *parent, int32_t x) {
    int32_t root = x;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[x] != root) {
        const int32_t next = parent[x];
        parent[x] = root;
        x = next;
    }
    return root;
}

static void uf_union_no_restrict(int32_t *parent, const int32_t x, const int32_t y) {
    const int32_t root_x = uf_find_no_restrict(parent, x);
    const int32_t root_y = uf_find_no_restrict(parent, y);
    if (root_x == root_y) {
        return;
    }
    if (root_x < root_y) {
        parent[root_y] = root_x;
    } else {
        parent[root_x] = root_y;
    }
}

CCResult *union_find_cc_no_restrict(const Graph *g) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    const int32_t num_vertices = graph_get_num_vertices(g);
    if (num_vertices <= 0) {
        fprintf(stderr, "Error: Invalid number of vertices\n");
        return NULL;
    }

    CCResult *result = malloc(sizeof(CCResult));
    if (result == NULL) {
        fprintf(stderr, "Error: Failed to allocate CCResult\n");
        return NULL;
    }

    int32_t *parent = malloc(sizeof(int32_t) * (size_t) num_vertices);
    if (parent == NULL) {
        fprintf(stderr, "Error: Failed to allocate parent array\n");
        free(result);
        return NULL;
    }

    for (int32_t i = 0; i < num_vertices; i++) {
        parent[i] = i;
    }

    for (int32_t v = 0; v < num_vertices; v++) {
        int32_t num_neighbors = 0;
        const int32_t *neighbors = graph_get_neighbors(g, v, &num_neighbors);
        if (neighbors == NULL) {
            continue;
        }
        for (int32_t j = 0; j < num_neighbors; j++) {
            const int32_t u = neighbors[j];
            uf_union_no_restrict(parent, v, u);
        }
    }

    result->labels = malloc(sizeof(int32_t) * (size_t) num_vertices);
    if (result->labels == NULL) {
        fprintf(stderr, "Error: Failed to allocate labels array\n");
        free(parent);
        free(result);
        return NULL;
    }

    for (int32_t i = 0; i < num_vertices; i++) {
        result->labels[i] = uf_find_no_restrict(parent, i);
    }

    result->num_iterations = 1;
    result->num_components = count_unique_labels(result->labels, num_vertices);
    if (result->num_components < 0) {
        fprintf(stderr, "Error: Failed to count components\n");
        free(parent);
        free(result->labels);
        free(result);
        return NULL;
    }

    free(parent);
    return result;
}

void cc_result_destroy(CCResult *result) {
    if (result == NULL) {
        return;
    }

    free(result->labels);
    free(result);
}

void cc_result_print_stats(const CCResult *result, const Graph *g) {
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
