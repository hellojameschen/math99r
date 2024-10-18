import numpy as np

def barycentric_refinement(adj_matrix):
    n = len(adj_matrix)  # number of vertices
    edges_in_triangles = set()
    triangles = []

    # Step 1: Find all triangles and triangle edges
    for i in range(n):
        for j in range(i+1, n):
            if adj_matrix[i][j]:  # i and j are connected
                for k in range(j+1, n):
                    if adj_matrix[i][k] and adj_matrix[j][k]:  # i-k and j-k are connected
                        triangles.append((i, j, k))
                        edges_in_triangles.add((i, j))
                        edges_in_triangles.add((i, k))
                        edges_in_triangles.add((j, k))

    # Step 2: Add midpoints for each edge
    new_adj_matrix = adj_matrix.copy()
    current_vertex = n  # Start counting new vertices after the original vertices
    edge_to_vertex = {}  # Map each edge to its midpoint vertex

    for edge in edges_in_triangles:
        i, j = edge

        # Add a new vertex at the midpoint of edge (i, j)
        midpoint_vertex = current_vertex
        current_vertex += 1

        # Remove the original edge (i, j)
        new_adj_matrix[i][j] = 0
        new_adj_matrix[j][i] = 0

        # Add new edges (i, midpoint) and (j, midpoint)
        new_adj_matrix = np.pad(new_adj_matrix, ((0, 1), (0, 1)), mode='constant')  # Expand matrix for new vertex
        new_adj_matrix[i][midpoint_vertex] = 1
        new_adj_matrix[midpoint_vertex][i] = 1
        new_adj_matrix[j][midpoint_vertex] = 1
        new_adj_matrix[midpoint_vertex][j] = 1

        # Store the mapping of edge to the midpoint vertex
        edge_to_vertex[edge] = midpoint_vertex

    # Step 3: Add centroid for each triangle and connect it to triangle vertices and midpoints
    for triangle in triangles:
        i, j, k = triangle

        # Add a new vertex for the centroid
        centroid_vertex = current_vertex
        current_vertex += 1

        # Add new row and column for the centroid in the adjacency matrix
        new_adj_matrix = np.pad(new_adj_matrix, ((0, 1), (0, 1)), mode='constant')

        # Connect the centroid to the three triangle vertices
        new_adj_matrix[centroid_vertex][i] = 1
        new_adj_matrix[i][centroid_vertex] = 1
        new_adj_matrix[centroid_vertex][j] = 1
        new_adj_matrix[j][centroid_vertex] = 1
        new_adj_matrix[centroid_vertex][k] = 1
        new_adj_matrix[k][centroid_vertex] = 1

        # Connect the centroid to the midpoints of the edges
        midpoint_ij = edge_to_vertex.get((i, j), edge_to_vertex.get((j, i)))
        midpoint_ik = edge_to_vertex.get((i, k), edge_to_vertex.get((k, i)))
        midpoint_jk = edge_to_vertex.get((j, k), edge_to_vertex.get((k, j)))

        new_adj_matrix[centroid_vertex][midpoint_ij] = 1
        new_adj_matrix[midpoint_ij][centroid_vertex] = 1
        new_adj_matrix[centroid_vertex][midpoint_ik] = 1
        new_adj_matrix[midpoint_ik][centroid_vertex] = 1
        new_adj_matrix[centroid_vertex][midpoint_jk] = 1
        new_adj_matrix[midpoint_jk][centroid_vertex] = 1

    return new_adj_matrix, current_vertex

# Step 4: Count the number of vertices, edges, and faces
def count_vertices_edges_faces(adj_matrix):
    num_vertices = len(adj_matrix)
    num_edges = np.sum(adj_matrix) // 2  # Count edges by summing half of the matrix
    num_faces = num_edges - num_vertices + 2  # Euler's formula for planar graphs: V - E + F = 2

    return num_vertices, num_edges, num_faces

# Step 5: Start with a triangle graph
adj_matrix_triangle = np.array([[0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 0]])

# Compute barycentric refinement
refined_adj_matrix, num_vertices = barycentric_refinement(adj_matrix_triangle)

# Count the vertices, edges, and faces in the refined graph
num_vertices, num_edges, num_faces = count_vertices_edges_faces(refined_adj_matrix)

# Display the results
print(f"Number of vertices: {num_vertices}")
print(f"Number of edges: {num_edges}")
print(f"Number of faces: {num_faces}")
