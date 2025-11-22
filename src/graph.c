#include "graph.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define INITIAL_CAPACITY 4

Graph* graph_create(const int32_t num_vertices) {
    if (num_vertices <= 0) {
        fprintf(stderr, "Error: Invalid number of vertices\n");
        return NULL;
    }

    Graph *g = (Graph *) malloc(sizeof(Graph));
    if (g == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for graph\n");
        return NULL;
    }

    g->num_vertices = num_vertices;
    g->num_edges = 0;
    g->finalized = false;

    /* Initialize CSR arrays as NULL (will be allocated during finalization) */
    g->row_ptr = NULL;
    g->col_idx = NULL;

    /* Allocate temporary storage for adjacency lists */
    g->adj_lists = (int32_t**)calloc((size_t)num_vertices, sizeof(int32_t*));
    g->degrees = (int32_t*)calloc((size_t)num_vertices, sizeof(int32_t));
    g->capacities = (int32_t*)calloc((size_t)num_vertices, sizeof(int32_t));

    if ((g->adj_lists == NULL) || (g->degrees == NULL) || (g->capacities == NULL)) {
        fprintf(stderr, "Error: Failed to allocate memory for adjacency lists\n");
        graph_destroy(g);
        return NULL;
    }

    /* Initialize each adjacency list with small capacity */
    for (int32_t i = 0; i < num_vertices; i++) {
        g->adj_lists[i] = (int32_t*)malloc(INITIAL_CAPACITY * sizeof(int32_t));
        if (g->adj_lists[i] == NULL) {
            fprintf(stderr, "Error: Failed to allocate adjacency list\n");
            graph_destroy(g);
            return NULL;
        }
        g->capacities[i] = INITIAL_CAPACITY;
        g->degrees[i] = 0;
    }

    return g;
}

int32_t graph_add_edge(Graph* g, const int32_t u, const int32_t v) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return -1;
    }

    if (g->finalized) {
        fprintf(stderr, "Error: Cannot add edges after finalization\n");
        return -1;
    }

    if ((u < 0) || (u >= g->num_vertices) || (v < 0) || (v >= g->num_vertices)) {
        fprintf(stderr, "Error: Invalid vertex indices: %d, %d\n", u, v);
        return -1;
    }

    /* Skip self-loops */
    if (u == v) {
        return 0;
    }

    /* Add v to u's adjacency list */
    if (g->degrees[u] >= g->capacities[u]) {
        /* Resize the adjacency list */
        g->capacities[u] *= 2;
        int32_t* temp = (int32_t*)realloc(g->adj_lists[u],
                                          (size_t)g->capacities[u] * sizeof(int32_t));
        if (temp == NULL) {
            fprintf(stderr, "Error: Failed to resize adjacency list\n");
            return -1;
        }
        g->adj_lists[u] = temp;
    }
    g->adj_lists[u][g->degrees[u]] = v;
    g->degrees[u]++;

    /* Add u to v's adjacency list (undirected graph) */
    if (g->degrees[v] >= g->capacities[v]) {
        g->capacities[v] *= 2;
        int32_t* temp = (int32_t*)realloc(g->adj_lists[v],
                                          (size_t)g->capacities[v] * sizeof(int32_t));
        if (temp == NULL) {
            fprintf(stderr, "Error: Failed to resize adjacency list\n");
            return -1;
        }
        g->adj_lists[v] = temp;
    }
    g->adj_lists[v][g->degrees[v]] = u;
    g->degrees[v]++;

    g->num_edges++;
    return 0;
}

int32_t graph_finalize(Graph* g) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return -1;
    }

    if (g->finalized) {
        return 0; /* Already finalized */
    }

    /* Allocate CSR arrays */
    g->row_ptr = (int32_t*)malloc((size_t)(g->num_vertices + 1) * sizeof(int32_t));
    g->col_idx = (int32_t*)malloc((size_t)(2 * g->num_edges) * sizeof(int32_t));

    if ((g->row_ptr == NULL) || (g->col_idx == NULL)) {
        fprintf(stderr, "Error: Failed to allocate CSR arrays\n");
        free(g->row_ptr);
        free(g->col_idx);
        g->row_ptr = NULL;
        g->col_idx = NULL;
        return -1;
    }

    /* Build row_ptr array */
    g->row_ptr[0] = 0;
    for (int32_t i = 0; i < g->num_vertices; i++) {
        g->row_ptr[i + 1] = g->row_ptr[i] + g->degrees[i];
    }

    /* Copy adjacency lists to col_idx */
    for (int32_t i = 0; i < g->num_vertices; i++) {
        const int32_t offset = g->row_ptr[i];
        (void)memcpy(&g->col_idx[offset], g->adj_lists[i],
                     (size_t)g->degrees[i] * sizeof(int32_t));
    }

    /* Free temporary storage */
    for (int32_t i = 0; i < g->num_vertices; i++) {
        free(g->adj_lists[i]);
    }
    free(g->adj_lists);
    free(g->degrees);
    free(g->capacities);

    g->adj_lists = NULL;
    g->degrees = NULL;
    g->capacities = NULL;
    g->finalized = true;

    return 0;
}

void graph_destroy(Graph* g) {
    if (g == NULL) {
        return;
    }

    /* Free CSR arrays if finalized */
    if (g->finalized) {
        free(g->row_ptr);
        free(g->col_idx);
    } else {
        /* Free temporary storage if not finalized */
        if (g->adj_lists != NULL) {
            for (int32_t i = 0; i < g->num_vertices; i++) {
                free(g->adj_lists[i]);
            }
            free(g->adj_lists);
        }
        free(g->degrees);
        free(g->capacities);
    }

    free(g);
}

int32_t graph_get_num_vertices(const Graph* g) {
    if (g == NULL) {
        return -1;
    }
    return g->num_vertices;
}

int32_t graph_get_num_edges(const Graph* g) {
    if (g == NULL) {
        return -1;
    }
    return g->num_edges;
}

const int32_t* graph_get_neighbors(const Graph* g, int32_t v, int32_t* num_neighbors) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return NULL;
    }

    if (num_neighbors == NULL) {
        fprintf(stderr, "Error: NULL num_neighbors pointer\n");
        return NULL;
    }

    if (!g->finalized) {
        fprintf(stderr, "Error: Graph must be finalized before accessing neighbors\n");
        return NULL;
    }

    if ((v < 0) || (v >= g->num_vertices)) {
        fprintf(stderr, "Error: Invalid vertex index: %d\n", v);
        return NULL;
    }

    *num_neighbors = g->row_ptr[v + 1] - g->row_ptr[v];
    return &g->col_idx[g->row_ptr[v]];
}

void graph_print_stats(const Graph* g) {
    if (g == NULL) {
        fprintf(stderr, "Error: NULL graph pointer\n");
        return;
    }

    printf("Graph Statistics:\n");
    printf("  Vertices: %d\n", g->num_vertices);
    printf("  Edges: %d\n", g->num_edges);
    printf("  Finalized: %s\n", g->finalized ? "Yes" : "No");

    if (g->finalized) {
        /* Calculate average degree */
        double avg_degree = (2.0 * (double)g->num_edges) / (double)g->num_vertices;
        printf("  Average degree: %.2f\n", avg_degree);

        /* Find min and max degree */
        int32_t min_degree = g->num_vertices;
        int32_t max_degree = 0;
        for (int32_t i = 0; i < g->num_vertices; i++) {
            const int32_t degree = g->row_ptr[i + 1] - g->row_ptr[i];
            if (degree < min_degree) {
                min_degree = degree;
            }
            if (degree > max_degree) {
                max_degree = degree;
            }
        }
        printf("  Min degree: %d\n", min_degree);
        printf("  Max degree: %d\n", max_degree);
    }
}
