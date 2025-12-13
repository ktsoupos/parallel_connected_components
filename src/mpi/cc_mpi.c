#include "cc_mpi.h"

#include <stdio.h>
#include <stdlib.h>

int partition_graph(const Graph *global_graph, DistributedGraph **dist_graph, MPI_Comm comm) {
    if (global_graph == NULL) {
        fprintf(stderr, "Error: NULL global_graph pointer\n");
        return -1;
    }
    if (dist_graph == NULL) {
        fprintf(stderr, "Error: NULL dist_graph pointer\n");
        return -1;
    }

    int rank, num_ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_ranks);

    // Only rank 0 should have a valid global_graph
    if (rank == 0) {
        fprintf(stderr, "Error: NULL global_graph pointer on rank 0\n");
        return -1;
    }

    int32_t metadata[2] = {0, 0};
    if (rank == 0) {
        metadata[0] = global_graph->num_vertices;
        metadata[1] = global_graph->num_edges;

    }
    MPI_Bcast(metadata, 2, MPI_INT32_T, 0, comm);
    const int32_t g_num_vertices = metadata[0];
    const int32_t g_num_edges = metadata[1];

    const int32_t verts_per_proc = g_num_vertices / num_ranks;
    const int32_t vertex_offset = rank * verts_per_proc;
    int32_t l_num_vertices;

    if (rank == num_ranks - 1) {
        // Last rank takes remaining vertices
        l_num_vertices = g_num_vertices - vertex_offset;
    } else {
        l_num_vertices = verts_per_proc;
    }

    *dist_graph = malloc(sizeof(DistributedGraph));
    if (*dist_graph == NULL) {
        fprintf(stderr, "Error: Failed to allocate DistributedGraph\n");
        return -1;
    }
    (*dist_graph)->g_num_vertices = g_num_vertices;
    (*dist_graph)->g_num_edges = g_num_edges;
    (*dist_graph)->l_num_vertices = l_num_vertices;
    (*dist_graph)->vertex_offset = vertex_offset;
    (*dist_graph)->rank = rank;
    (*dist_graph)->num_ranks = num_ranks;
    (*dist_graph)->comm = comm;

    (*dist_graph)->local_row_ptr = malloc(sizeof(int32_t) * (size_t)(l_num_vertices + 1));
    if ((*dist_graph)->local_row_ptr == NULL) {
        fprintf(stderr, "Error: Failed to allocate local row ptr\n");
        free(*dist_graph);
        return -1;
    }

    int *sendcounts = malloc(sizeof(int) * (size_t)num_ranks);
    if (sendcounts == NULL) {
        fprintf(stderr, "Error: Failed to allocate sendcounts\n");
        free((*dist_graph)->local_row_ptr);
        free(*dist_graph);
        return -1;
    }

    int *displs = malloc(sizeof(int) * (size_t)num_ranks);
    if (displs == NULL) {
        fprintf(stderr, "Error: Failed to allocate displs\n");
        free(sendcounts);
        free((*dist_graph)->local_row_ptr);
        free(*dist_graph);
        return -1;
    }

    if (rank == 0) {
        for (int i = 0; i < num_ranks; i++) {
            const int offset = i * verts_per_proc;
            const int count = (i == num_ranks - 1)
                                  ? (g_num_vertices - offset + 1)
                                  : (verts_per_proc + 1);
            sendcounts[i] = count;
            displs[i] = offset;
        }
    }
    // clang-format off
    MPI_Scatterv(
        (rank == 0) ? global_graph->row_ptr : NULL,  // sendbuf (only rank 0)
        sendcounts,                                // send counts per process
        displs,                                    // displacements
        MPI_INT32_T,                               // sendtype
        (*dist_graph)->local_row_ptr,              // recvbuf
        l_num_vertices + 1,                        // recvcount
        MPI_INT32_T,                               // recvtype
        0,                                         // root
        comm
    );
    // clang-format on

    //The row_ptr values need to be adjusted to start from 0:
    const int32_t offset_adjustment = (*dist_graph)->local_row_ptr[0];
    for (int32_t i = 0; i <= l_num_vertices; i++) {
        (*dist_graph)->local_row_ptr[i] -= offset_adjustment;
    }

    (*dist_graph)->l_num_edges = (*dist_graph)->local_row_ptr[l_num_vertices];

    (*dist_graph)->local_col_idx = malloc(sizeof(int32_t) * (size_t)(*dist_graph)->l_num_edges);
    if ((*dist_graph)->local_col_idx == NULL) {
        fprintf(stderr, "Error: Failed to allocate local col idx\n");
        free((*dist_graph)->local_row_ptr);
        free(*dist_graph);
        free(sendcounts);
        free(displs);
        return -1;
    }

    // On rank 0: calculate edge counts and offsets for each process
    if (rank == 0) {
        for (int i = 0; i < num_ranks; i++) {
            const int v_offset = i * verts_per_proc;
            const int v_count = (i == num_ranks - 1)
                              ? (g_num_vertices - v_offset)
                              : verts_per_proc;

            const int edge_start = global_graph->row_ptr[v_offset];
            const int edge_end = global_graph->row_ptr[v_offset + v_count];

            sendcounts[i] = edge_end - edge_start;
            displs[i] = edge_start;
        }
    }

    MPI_Scatterv(
        rank == 0 ? global_graph->col_idx : NULL,
        sendcounts,
        displs,
        MPI_INT32_T,
        (*dist_graph)->local_col_idx,
        (*dist_graph)->l_num_edges,
        MPI_INT32_T,
        0,
        comm
        );

    // Cleanup temporary arrays
    free(sendcounts);
    free(displs);

    return 0;
}