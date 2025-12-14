#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Generate a random graph in Matrix Market format
 * Creates multiple components with random edges
 */
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <num_vertices> <num_components> <edges_per_vertex>\n", argv[0]);
        fprintf(stderr, "Example: %s 10000 10 8\n", argv[0]);
        return 1;
    }

    const int num_vertices = atoi(argv[1]);
    const int num_components = atoi(argv[2]);
    const int edges_per_vertex = atoi(argv[3]);

    if (num_vertices < num_components || num_components < 1) {
        fprintf(stderr, "Error: Invalid parameters\n");
        return 1;
    }

    srand((unsigned int)time(NULL));

    // Calculate vertices per component
    const int verts_per_comp = num_vertices / num_components;

    // Each vertex will have approximately edges_per_vertex edges
    // But we create undirected edges, so divide by 2
    const int num_edges = (num_vertices * edges_per_vertex) / 2;

    // Print Matrix Market header
    printf("%%%%MatrixMarket matrix coordinate pattern symmetric\n");
    printf("%% Generated graph: %d vertices, ~%d components, ~%d edges\n",
           num_vertices, num_components, num_edges);
    printf("%d %d %d\n", num_vertices, num_vertices, num_edges);

    int edges_created = 0;

    // Create components by connecting vertices within each component
    for (int comp = 0; comp < num_components; comp++) {
        const int start = comp * verts_per_comp;
        const int end = (comp == num_components - 1) ? num_vertices : (comp + 1) * verts_per_comp;
        const int comp_size = end - start;

        // Create a connected component (tree structure first)
        for (int i = start + 1; i < end; i++) {
            // Connect to a random earlier vertex in same component
            const int target = start + (rand() % (i - start));
            printf("%d %d\n", i + 1, target + 1);  // MTX is 1-indexed
            edges_created++;
        }

        // Add random edges within component
        const int extra_edges = (comp_size * edges_per_vertex) / 2 - comp_size;
        for (int e = 0; e < extra_edges && edges_created < num_edges; e++) {
            const int u = start + (rand() % comp_size);
            const int v = start + (rand() % comp_size);
            if (u != v) {
                printf("%d %d\n", u + 1, v + 1);
                edges_created++;
            }
        }
    }

    // Fill remaining edges if needed
    while (edges_created < num_edges) {
        // Random edge within same component
        const int comp = rand() % num_components;
        const int start = comp * verts_per_comp;
        const int end = (comp == num_components - 1) ? num_vertices : (comp + 1) * verts_per_comp;
        const int comp_size = end - start;

        const int u = start + (rand() % comp_size);
        const int v = start + (rand() % comp_size);
        if (u != v) {
            printf("%d %d\n", u + 1, v + 1);
            edges_created++;
        }
    }

    return 0;
}
