#include "mtx_reader.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024

/**
 * Skip comment lines and whitespace in MTX file
 * Returns: 0 on success, -1 on error
 */
static int32_t skip_comments(FILE *file) {
    char line[MAX_LINE_LENGTH];

    while (fgets(line, sizeof(line), file) != NULL) {
        /* Skip empty lines and whitespace */
        const char *ptr = line;
        while (*ptr && isspace((unsigned char)*ptr)) {
            ptr++;
        }

        /* If line is not a comment, rewind and return */
        if (*ptr != '%' && *ptr != '\0') {
            const long position = ftell(file);
            if (position == -1) {
                return -1;
            }
            /* Rewind to start of this line */
            if (fseek(file, position - (long)strlen(line), SEEK_SET) != 0) {
                return -1;
            }
            return 0;
        }
    }

    return -1; /* No non-comment line found */
}

/**
 * Check if MTX header indicates symmetric matrix
 * Returns: true if symmetric, false otherwise
 */
static bool is_symmetric_matrix(FILE *file) {
    char line[MAX_LINE_LENGTH];
    bool is_symmetric = false;

    /* Save current position */
    long position = ftell(file);
    if (position == -1) {
        return false;
    }

    /* Rewind to beginning */
    rewind(file);

    /* Read first line (header) */
    if (fgets(line, sizeof(line), file) != NULL) {
        /* Convert to lowercase for comparison */
        for (char *p = line; *p; p++) {
            *p = (char)tolower((unsigned char)*p);
        }

        /* Check if "symmetric" appears in header */
        if (strstr(line, "symmetric") != NULL) {
            is_symmetric = true;
        }
    }

    /* Restore position */
    if (fseek(file, position, SEEK_SET) != 0) {
        return false;
    }

    return is_symmetric;
}

Graph *read_mtx_file(const char *filename) {
    return read_mtx_file_verbose(filename, 0);
}

Graph *read_mtx_file_verbose(const char *filename, const int32_t report_interval) {
    if (filename == NULL) {
        fprintf(stderr, "Error: NULL filename\n");
        return NULL;
    }

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return NULL;
    }

    /* Check if matrix is symmetric */
    const bool is_symmetric = is_symmetric_matrix(file);

    /* Skip comments to get to dimensions */
    if (skip_comments(file) != 0) {
        fprintf(stderr, "Error: Failed to read MTX file header\n");
        fclose(file);
        return NULL;
    }

    /* Read matrix dimensions: num_rows num_cols num_entries */
    char line[MAX_LINE_LENGTH];
    if (fgets(line, sizeof(line), file) == NULL) {
        fprintf(stderr, "Error: Failed to read matrix dimensions\n");
        fclose(file);
        return NULL;
    }

    int32_t num_rows, num_cols, num_entries;
    if (sscanf(line, "%d %d %d", &num_rows, &num_cols, &num_entries) != 3) {
        fprintf(stderr, "Error: Failed to parse matrix dimensions\n");
        fclose(file);
        return NULL;
    }

    /* For graph, rows and cols should be equal */
    if (num_rows != num_cols) {
        fprintf(stderr, "Warning: Non-square matrix (%d x %d), using max dimension\n", num_rows,
                num_cols);
    }

    const int32_t num_vertices = (num_rows > num_cols) ? num_rows : num_cols;

    if (report_interval > 0) {
        printf("Reading MTX file: %s\n", filename);
        printf("  Vertices: %d\n", num_vertices);
        printf("  Entries: %d\n", num_entries);
        printf("  Symmetric: %s\n", is_symmetric ? "Yes" : "No");
    }

    /* Create graph */
    Graph *g = graph_create(num_vertices);
    if (g == NULL) {
        fprintf(stderr, "Error: Failed to create graph\n");
        fclose(file);
        return NULL;
    }

    /* Read edges */
    int32_t edges_read = 0;
    for (int32_t i = 0; i < num_entries; i++) {
        if (fgets(line, sizeof(line), file) == NULL) {
            fprintf(stderr, "Error: Failed to read edge %d\n", i);
            graph_destroy(g);
            fclose(file);
            return NULL;
        }

        int32_t u, v;
        double value;

        /* Try to parse edge with optional value */
        int32_t fields = sscanf(line, "%d %d %lf", &u, &v, &value);

        if (fields < 2) {
            fprintf(stderr, "Error: Failed to parse edge %d\n", i);
            graph_destroy(g);
            fclose(file);
            return NULL;
        }

        /* Convert to 0-indexed (MTX files are typically 1-indexed) */
        u--;
        v--;

        /* Validate indices */
        if (u < 0 || u >= num_vertices || v < 0 || v >= num_vertices) {
            fprintf(stderr, "Error: Invalid edge indices at line %d: (%d, %d)\n", i + 1, u + 1,
                    v + 1);
            graph_destroy(g);
            fclose(file);
            return NULL;
        }

        /* Add edge (graph_add_edge handles undirected edges) */
        if (graph_add_edge(g, u, v) != 0) {
            fprintf(stderr, "Error: Failed to add edge (%d, %d)\n", u, v);
            graph_destroy(g);
            fclose(file);
            return NULL;
        }

        edges_read++;

        /* Report progress */
        if (report_interval > 0 && (edges_read % report_interval == 0)) {
            printf("  Read %d / %d edges (%.1f%%)\n", edges_read, num_entries,
                   100.0 * edges_read / num_entries);
        }
    }

    fclose(file);

    if (report_interval > 0) {
        printf("  Finalizing graph...\n");
    }

    /* Finalize graph (convert to CSR format) */
    if (graph_finalize(g) != 0) {
        fprintf(stderr, "Error: Failed to finalize graph\n");
        graph_destroy(g);
        return NULL;
    }

    if (report_interval > 0) {
        printf("  Done!\n");
        graph_print_stats(g);
    }

    return g;
}
