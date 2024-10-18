import pygame
import math
import sys
import heapq  # For priority queue in Dijkstra's algorithm
import random
import numpy as np

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 750  # Screen dimensions
PANEL_HEIGHT = 200        # Height reserved for the UI panel
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Force-Directed Graph with Octahedron Initialization")

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
LIGHT_BLUE = (173, 216, 230)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)      # Color for highlighted edges
PURPLE = (128, 0, 128)      # Distinct color for the source node
BUTTON_COLOR = (70, 130, 180)        # Steel Blue for buttons
BUTTON_HOVER_COLOR = (100, 149, 237) # Cornflower Blue for button hover
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Slider Parameters
MIN_SLIDER = 1
MAX_SLIDER = 20
SLIDER_WIDTH = 200
SLIDER_HEIGHT = 20
slider_value = 5  # Initial path length

# Additional Slider Parameters for Barycentric Refinement and Shrink Nodes
# (Optional: Adjust as needed)
REFINE_BUTTON_WIDTH = 100
REFINE_BUTTON_HEIGHT = 30
SHRINK_BUTTON_WIDTH = 100
SHRINK_BUTTON_HEIGHT = 30

# Physics Parameters
REPULSION_STRENGTH = 5000
ATTRACTION_STRENGTH = 0.01
DAMPING = 0.9
TIME_STEP = 0.1

# Fonts
FONT = pygame.font.SysFont(None, 24)
FONT_SMALL = pygame.font.SysFont(None, 20)

# Node Class with Edges
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.fx = 0.0
        self.fy = 0.0
        self.fixed = False
        self.edges = []

        # Attributes for Dijkstra's traversal
        self.visited = False
        self.distance = float('inf')  # Tentative distance from source
        self.previous = []  # List of previous nodes in the optimal paths
        self.in_queue = False  # Indicates if the node is in the priority queue

# Slider Class
class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, height)
        self.min = min_val
        self.max = max_val
        self.value = initial_val
        self.handle_radius = height // 2
        self.handle_x = self.get_handle_x()
        self.dragging = False  # Initialize dragging state
        self.label = label

    def get_handle_x(self):
        proportion = (self.value - self.min) / (self.max - self.min)
        return self.rect.x + int(proportion * self.rect.width)

    def draw(self, surface):
        # Draw slider line
        pygame.draw.line(surface, BLACK, (self.rect.x, self.rect.centery),
                         (self.rect.x + self.rect.width, self.rect.centery), 2)
        # Draw handle
        pygame.draw.circle(surface, BLUE, (self.handle_x, self.rect.centery), self.handle_radius)

        # Draw label
        label_surface = FONT.render(f"{self.label}: {self.value}", True, BLACK)
        surface.blit(label_surface, (self.rect.x, self.rect.y - 25))

    def handle_event(self, event):
        global slider_value
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            distance = math.hypot(mouse_x - self.handle_x, mouse_y - self.rect.centery)
            if distance <= self.handle_radius:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mouse_x, _ = event.pos
                # Clamp handle position
                self.handle_x = max(self.rect.x, min(mouse_x, self.rect.x + self.rect.width))
                # Update value based on handle position
                proportion = (self.handle_x - self.rect.x) / self.rect.width
                self.value = int(self.min + proportion * (self.max - self.min))
                slider_value = self.value

# Button Class
class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.hovered = False

    def draw(self, surface):
        color = BUTTON_HOVER_COLOR if self.hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect)
        # Draw text
        text_surface = FONT.render(self.text, True, WHITE)
        surface.blit(text_surface, (
            self.rect.x + (self.rect.width - text_surface.get_width()) // 2,
            self.rect.y + (self.rect.height - text_surface.get_height()) // 2
        ))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            mouse_x, mouse_y = event.pos
            self.hovered = self.rect.collidepoint((mouse_x, mouse_y))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            if self.rect.collidepoint((mouse_x, mouse_y)):
                return True  # Button was clicked
        return False

# Generate Octahedron Graph
def generate_octahedron_graph():
    nodes = []
    adj_matrix = [[0]*6 for _ in range(6)]  # 6 nodes for an octahedron

    # Define positions for an octahedron centered on the screen
    center_x, center_y = WIDTH // 2, (HEIGHT - PANEL_HEIGHT) // 2
    radius = 150  # Distance from center to each vertex

    # Octahedron has 6 vertices: one top, one bottom, and four around the center
    positions = [
        (center_x, center_y - radius),          # Top node (0)
        (center_x, center_y + radius),          # Bottom node (1)
        (center_x - radius, center_y),          # Left node (2)
        (center_x + radius, center_y),          # Right node (3)
        (center_x, center_y - radius // 2),      # Upper-middle node (4)
        (center_x, center_y + radius // 2)       # Lower-middle node (5)
    ]

    # Create nodes and set positions
    for pos in positions:
        nodes.append(Node(*pos))

    # Define edges for an octahedron (connecting vertices to form 8 triangular faces)
    edges = [
        (0, 2), (0, 4), (0, 3), (0, 5),  # Top vertex connected to middle vertices
        (1, 2), (1, 4), (1, 3), (1, 5),  # Bottom vertex connected to middle vertices
        (2, 4), (4, 3), (3, 5), (5, 2)   # Middle vertices forming square around the center
    ]

    # Populate adjacency matrix and edges list
    for (i, j) in edges:
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1  # Undirected graph
        nodes[i].edges.append(j)
        nodes[j].edges.append(i)

    return nodes, adj_matrix

# Apply Forces to Nodes
def apply_forces(nodes, adj_matrix):
    num_nodes = len(nodes)
    # Reset forces
    for node in nodes:
        node.fx = 0.0
        node.fy = 0.0

    # Repulsion between nodes
    for i in range(num_nodes):
        node1 = nodes[i]
        if node1.fixed:
            continue  # Skip force calculations for fixed nodes
        for j in range(i + 1, num_nodes):
            node2 = nodes[j]
            if node2.fixed:
                continue  # Skip force calculations for fixed nodes
            dx = node1.x - node2.x
            dy = node1.y - node2.y
            distance_sq = dx * dx + dy * dy
            if distance_sq == 0:
                continue  # Prevent division by zero
            repulsion_force = REPULSION_STRENGTH / distance_sq
            distance = math.sqrt(distance_sq)
            fx = (dx / distance) * repulsion_force
            fy = (dy / distance) * repulsion_force
            node1.fx += fx
            node1.fy += fy
            node2.fx -= fx  # Newton's third law
            node2.fy -= fy

    # Attraction along edges
    for i in range(num_nodes):
        node = nodes[i]
        if node.fixed:
            continue  # Skip force calculations for fixed nodes
        for j in node.edges:
            other_node = nodes[j]
            dx = other_node.x - node.x
            dy = other_node.y - node.y
            distance = math.hypot(dx, dy)
            if distance == 0:
                continue  # Prevent division by zero
            attraction_force = ATTRACTION_STRENGTH * distance
            fx = (dx / distance) * attraction_force
            fy = (dy / distance) * attraction_force
            node.fx += fx
            node.fy += fy
            other_node.fx -= fx  # Newton's third law
            other_node.fy -= fy

    # Update positions based on forces
    for node in nodes:
        if node.fixed:
            node.vx = 0.0
            node.vy = 0.0
            continue  # Skip position update for fixed nodes
        node.vx = (node.vx + node.fx * TIME_STEP) * DAMPING
        node.vy = (node.vy + node.fy * TIME_STEP) * DAMPING
        node.x += node.vx * TIME_STEP
        node.y += node.vy * TIME_STEP

# Compute Total Potential Energy
def compute_energy(nodes, adj_matrix):
    total_energy = 0.0
    num_nodes = len(nodes)

    # Repulsive energy between all node pairs
    for i in range(num_nodes):
        node1 = nodes[i]
        for j in range(i + 1, num_nodes):
            node2 = nodes[j]
            dx = node1.x - node2.x
            dy = node1.y - node2.y
            distance = math.hypot(dx, dy)
            if distance == 0:
                continue  # Prevent division by zero
            repulsion_energy = REPULSION_STRENGTH / distance
            total_energy += repulsion_energy

    # Attractive energy along edges
    for i in range(num_nodes):
        node = nodes[i]
        for j in node.edges:
            if j > i:  # To avoid double-counting edges
                other_node = nodes[j]
                dx = other_node.x - node.x
                dy = other_node.y - node.y
                distance = math.hypot(dx, dy)
                if distance == 0:
                    continue  # Prevent division by zero
                attraction_energy = ATTRACTION_STRENGTH * distance * distance / 2
                total_energy += attraction_energy

    return total_energy

# Get Ordered Neighbors by Cycle
def get_ordered_neighbors_by_cycle(adj_matrix, node_index):
    """
    Returns the neighbors of the specified node ordered cyclically based on their connections.

    :param adj_matrix: Adjacency matrix representing the graph.
    :param node_index: Index of the node whose neighbors are to be ordered.
    :return: List of neighbor indices ordered cyclically. Returns empty list if not a cycle.
    """
    neighbors = [i for i, connected in enumerate(adj_matrix[node_index]) if connected]

    if not neighbors:
        return []

    # Build adjacency list for the neighbors
    neighbor_adj = {n: [] for n in neighbors}

    for n in neighbors:
        for m in neighbors:
            if adj_matrix[n][m] and m != n:
                neighbor_adj[n].append(m)

    # Verify that each neighbor has exactly two connections (cycle property)
    for n, connections in neighbor_adj.items():
        if len(connections) != 2:
            # The subgraph is not a single cycle
            return []

    # Traverse the cycle
    ordered = []
    visited = set()

    # Start traversal from the first neighbor
    current = neighbors[0]
    prev = None

    while len(ordered) < len(neighbors):
        ordered.append(current)
        visited.add(current)

        # Get connected neighbors excluding the previous node to prevent backtracking
        connections = neighbor_adj[current]
        next_nodes = [n for n in connections if n != prev]

        if not next_nodes:
            # Dead end reached, cycle cannot be completed
            return []

        next_node = next_nodes[0]
        prev, current = current, next_node

    return ordered

# Get Ordered Neighbors by Angle (Fallback)
def get_ordered_neighbors(nodes, adj_matrix, node_index):
    """
    Returns the neighbors of the specified node ordered by their geometric angle.

    :param nodes: List of Node objects.
    :param adj_matrix: Adjacency matrix representing the graph.
    :param node_index: Index of the node whose neighbors are to be ordered.
    :return: List of neighbor indices ordered by angle.
    """
    node = nodes[node_index]

    # Step 1: Identify all neighbors of the node
    neighbors = [i for i, connected in enumerate(adj_matrix[node_index]) if connected]

    if not neighbors:
        return []

    # Step 2: Calculate the angle of each neighbor relative to the node
    angles = []
    for neighbor in neighbors:
        neighbor_node = nodes[neighbor]
        dx = neighbor_node.x - node.x
        dy = neighbor_node.y - node.y
        angle = math.atan2(dy, dx)
        angles.append((angle, neighbor))

    # Step 3: Sort neighbors by angle to achieve cyclic ordering
    angles.sort()
    ordered_neighbors = [neighbor for angle, neighbor in angles]

    return ordered_neighbors

def generate_refined_graph(nodes, adj_matrix, num_refinements=1):
    """
    Performs barycentric refinement on the graph by subdividing existing triangles.

    :param nodes: List of Node objects.
    :param adj_matrix: Adjacency matrix representing the graph.
    :param num_refinements: Number of times to perform the refinement.
    :return: Updated nodes, adj_matrix after refinement.
    """
    import numpy as np
    n = len(nodes)  # number of original nodes

    for _ in range(num_refinements):
        edges_in_triangles = set()
        triangles = []

        # Step 1: Find all triangles and their edges
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j]:  # i and j are connected
                    for k in range(j + 1, n):
                        if adj_matrix[i][k] and adj_matrix[j][k]:  # i-k and j-k are connected
                            triangles.append((i, j, k))
                            edges_in_triangles.add((i, j))
                            edges_in_triangles.add((i, k))
                            edges_in_triangles.add((j, k))

        # Step 2: Add midpoints for each edge
        new_adj_matrix = adj_matrix.copy()
        current_vertex = len(nodes)  # Start counting new vertices after the original vertices
        edge_to_vertex = {}  # Map each edge to its midpoint vertex

        for edge in edges_in_triangles:
            i, j = edge

            # Calculate midpoint position
            x_mid = (nodes[i].x + nodes[j].x) / 2
            y_mid = (nodes[i].y + nodes[j].y) / 2

            # Add a new vertex (Node) at the midpoint of edge (i, j)
            midpoint_vertex = current_vertex
            midpoint_node = Node(x_mid, y_mid)
            nodes.append(midpoint_node)
            current_vertex += 1

            # Remove the original edge (i, j)
            new_adj_matrix[i][j] = 0
            new_adj_matrix[j][i] = 0

            if j in nodes[i].edges:
                nodes[i].edges.remove(j)  # Remove j from i's edges list
            if i in nodes[j].edges:
                nodes[j].edges.remove(i)  # Remove i from j's edges list

            # Add new edges (i, midpoint) and (j, midpoint)
            new_adj_matrix = np.pad(new_adj_matrix, ((0, 1), (0, 1)), mode='constant')  # Expand matrix for new vertex
            new_adj_matrix[i][midpoint_vertex] = 1
            new_adj_matrix[midpoint_vertex][i] = 1
            new_adj_matrix[j][midpoint_vertex] = 1
            new_adj_matrix[midpoint_vertex][j] = 1

            # Update edges list for the original nodes and new midpoint node
            nodes[i].edges.append(midpoint_vertex)
            nodes[j].edges.append(midpoint_vertex)
            midpoint_node.edges.append(i)
            midpoint_node.edges.append(j)

            # Store the mapping of edge to the midpoint vertex
            edge_to_vertex[edge] = midpoint_vertex

        # Step 3: Add centroid for each triangle and connect it to triangle vertices and midpoints
        for triangle in triangles:
            i, j, k = triangle

            # Calculate centroid position
            x_centroid = (nodes[i].x + nodes[j].x + nodes[k].x) / 3
            y_centroid = (nodes[i].y + nodes[j].y + nodes[k].y) / 3

            # Add a new vertex (Node) for the centroid
            centroid_vertex = current_vertex
            centroid_node = Node(x_centroid, y_centroid)
            nodes.append(centroid_node)
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

            # Update edges list for the triangle vertices and the centroid node
            nodes[i].edges.append(centroid_vertex)
            nodes[j].edges.append(centroid_vertex)
            nodes[k].edges.append(centroid_vertex)
            centroid_node.edges.append(i)
            centroid_node.edges.append(j)
            centroid_node.edges.append(k)

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

            # Update edges list for the centroid node and midpoints
            centroid_node.edges.append(midpoint_ij)
            centroid_node.edges.append(midpoint_ik)
            centroid_node.edges.append(midpoint_jk)
            nodes[midpoint_ij].edges.append(centroid_vertex)
            nodes[midpoint_ik].edges.append(centroid_vertex)
            nodes[midpoint_jk].edges.append(centroid_vertex)

    return nodes, new_adj_matrix





# Collapse Nodes Function
def collapse_nodes(nodes, adj_matrix, num_pairs):
    """
    Collapses specified pairs of nodes into single nodes to simplify the graph.

    :param nodes: List of Node objects.
    :param adj_matrix: Adjacency matrix representing the graph.
    :param num_pairs: Number of node pairs to collapse.
    :return: Updated nodes, adj_matrix, and list of collapsed pairs.
    """
    num_nodes = len(nodes)
    if num_pairs <= 0:
        print("Number of pairs to collapse must be positive.")
        return nodes, adj_matrix, []
    if 2 * num_pairs > num_nodes:
        print("Not enough nodes to collapse the specified number of pairs.")
        return nodes, adj_matrix, []

    # Randomly select 2 * num_pairs unique node indices
    available_indices = list(range(num_nodes))
    random.shuffle(available_indices)
    selected_indices = available_indices[:2 * num_pairs]

    # Pair them
    pairs = [(selected_indices[i], selected_indices[i + 1]) for i in range(0, 2 * num_pairs, 2)]

    # To keep track of which nodes have been collapsed
    collapsed_nodes = []

    for a_idx, b_idx in pairs:
        node_a = nodes[a_idx]
        node_b = nodes[b_idx]

        # Create a new node at the average position
        new_x = (node_a.x + node_b.x) / 2
        new_y = (node_a.y + node_b.y) / 2
        new_node = Node(new_x, new_y)
        nodes.append(new_node)

        # Merge edges
        new_edges = set(node_a.edges + node_b.edges)
        new_edges.discard(a_idx)
        new_edges.discard(b_idx)
        new_edges = list(new_edges)
        new_node.edges = new_edges

        # Update adjacency matrix
        adj_matrix.append([0] * len(nodes))  # New row for new node
        for row in adj_matrix:
            row.append(0)  # New column for new node

        new_node_idx = len(nodes) - 1
        for neighbor_idx in new_edges:
            adj_matrix[new_node_idx][neighbor_idx] = 1
            adj_matrix[neighbor_idx][new_node_idx] = 1
            nodes[neighbor_idx].edges.append(new_node_idx)
            # Remove old connections
            if a_idx in nodes[neighbor_idx].edges:
                nodes[neighbor_idx].edges.remove(a_idx)
            if b_idx in nodes[neighbor_idx].edges:
                nodes[neighbor_idx].edges.remove(b_idx)

        # Remove nodes a and b from the graph
        # First, remove their connections
        for neighbor_idx in node_a.edges:
            adj_matrix[a_idx][neighbor_idx] = 0
            adj_matrix[neighbor_idx][a_idx] = 0
            if a_idx in nodes[neighbor_idx].edges:
                nodes[neighbor_idx].edges.remove(a_idx)
        for neighbor_idx in node_b.edges:
            adj_matrix[b_idx][neighbor_idx] = 0
            adj_matrix[neighbor_idx][b_idx] = 0
            if b_idx in nodes[neighbor_idx].edges:
                nodes[neighbor_idx].edges.remove(b_idx)

        # Mark nodes a and b for removal
        collapsed_nodes.extend([a_idx, b_idx])

    # Remove collapsed nodes from the nodes list and adjacency matrix
    # To avoid issues with shifting indices, remove nodes in descending order
    collapsed_nodes = sorted(collapsed_nodes, reverse=True)
    for idx in collapsed_nodes:
        del nodes[idx]
        del adj_matrix[idx]
    # Remove corresponding columns
    for row in adj_matrix:
        for idx in collapsed_nodes:
            del row[idx]

    # Update edge indices in nodes after removal
    index_mapping = {}
    current_idx = 0
    for original_idx in range(len(nodes) + len(collapsed_nodes)):
        if original_idx not in collapsed_nodes:
            index_mapping[original_idx] = current_idx
            current_idx += 1
    for node in nodes:
        updated_edges = []
        for edge in node.edges:
            updated_edges.append(index_mapping[edge])
        node.edges = list(set(updated_edges))  # Remove duplicate edges

    print(f"Collapsed {num_pairs} pairs of nodes.")
    return nodes, adj_matrix, pairs

# Find Triangles in the Graph
def find_triangles(nodes, adj_matrix):
    """
    Identifies all unique triangles in the current graph.

    :param nodes: List of Node objects.
    :param adj_matrix: Adjacency matrix representing the graph.
    :return: List of triangles, where each triangle is a tuple of node indices.
    """
    triangles = []
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in nodes[i].edges:
            if j > i:
                for k in nodes[j].edges:
                    if k > j and adj_matrix[i][k]:
                        triangles.append((i, j, k))
    return triangles

def find_neighbors(adjacency_matrix, node):
    """Find all neighbors of a given node from the adjacency matrix."""
    return [i for i, connected in enumerate(adjacency_matrix[node]) if connected]

def find_neighbor_cycle(adjacency_matrix, node):
    """Order neighbors of the given node based on the cycle formed by edges between them."""
    neighbors = find_neighbors(adjacency_matrix, node)
    
    if len(neighbors) < 2:
        return neighbors  # No cycle possible if fewer than 2 neighbors
    
    cycle = [neighbors[0]]  # Start from the first neighbor
    visited = set(cycle)
    
    while len(cycle) < len(neighbors):
        current = cycle[-1]  # Last node in the cycle
        for neighbor in neighbors:
            if adjacency_matrix[current][neighbor] and neighbor not in visited:
                cycle.append(neighbor)
                visited.add(neighbor)
                break
    
    return cycle

def find_antipodal_neighbor(adjacency_matrix, first_vertex, second_vertex):
    """Find the antipodal neighbor of first_vertex with respect to second_vertex."""
    neighbors_of_second = find_neighbor_cycle(adjacency_matrix, second_vertex)
    first_index = neighbors_of_second.index(first_vertex)
    antipodal_index = (first_index + len(neighbors_of_second) // 2) % len(neighbors_of_second)
    return neighbors_of_second[antipodal_index]

def find_antipodal_path(adjacency_matrix, start_vertex, next_vertex, steps):
    """Recursively find a path by continually finding the next antipodal vertex."""
    path = [start_vertex, next_vertex]  # Start with the first two vertices
    
    for _ in range(steps - 2):  # Already have 2 vertices in the path, so do steps-2
        new_vertex = find_antipodal_neighbor(adjacency_matrix, path[-2], path[-1])
        path.append(new_vertex)
    
    return path

# Draw Traversal Path
def draw_traversal_path(screen, nodes, path, zoom, offset_x, offset_y):
    if len(path) < 2:
        return
    for i in range(len(path) - 1):
        node_a = nodes[path[i]]
        node_b = nodes[path[i + 1]]
        x1 = int((node_a.x + offset_x) * zoom)
        y1 = int((node_a.y + offset_y) * zoom)
        x2 = int((node_b.x + offset_x) * zoom)
        y2 = int((node_b.y + offset_y) * zoom)
        pygame.draw.line(screen, RED, (x1, y1), (x2, y2), 2)

    # Highlight the nodes in the path
    for node_index in path:
        node = nodes[node_index]
        x = (node.x + offset_x) * zoom
        y = (node.y + offset_y) * zoom
        pygame.draw.circle(screen, GREEN, (int(x), int(y)), 8)

# Draw the Graph
def draw_graph(screen, nodes, adj_matrix, zoom, offset_x, offset_y, dragged_node):
    # Define the drawing area (excluding the panel)
    drawing_area_rect = pygame.Rect(0, 0, WIDTH, HEIGHT - PANEL_HEIGHT)
    pygame.draw.rect(screen, WHITE, drawing_area_rect)

    num_nodes = len(nodes)

    # Draw edges
    for i in range(num_nodes):
        node = nodes[i]
        for j in node.edges:
            if j > i:  # To avoid drawing the same edge twice
                other_node = nodes[j]
                # Apply zoom and offset
                x1 = int((node.x + offset_x) * zoom)
                y1 = int((node.y + offset_y) * zoom)
                x2 = int((other_node.x + offset_x) * zoom)
                y2 = int((other_node.y + offset_y) * zoom)
                pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), 1)

    # Draw nodes
    for idx, node in enumerate(nodes):
        # Apply zoom and offset
        x = (node.x + offset_x) * zoom
        y = (node.y + offset_y) * zoom
        # Only draw nodes that are within the visible screen area
        if 0 <= x <= WIDTH and 0 <= y <= (HEIGHT - PANEL_HEIGHT):
            color = GRAY
            if node == dragged_node:
                color = RED  # Highlight dragged node
            pygame.draw.circle(screen, color, (int(x), int(y)), 6 if node == dragged_node else 4)

# Draw UI Panel
def draw_ui(screen, sliders, buttons, energy, shrink_input_text, num_vertices, num_edges):
    # Define the panel area
    panel_rect = pygame.Rect(0, HEIGHT - PANEL_HEIGHT, WIDTH, PANEL_HEIGHT)
    pygame.draw.rect(screen, DARK_GRAY, panel_rect)

    # Draw Sliders
    for slider in sliders:
        slider.draw(screen)

    # Draw Buttons
    for button in buttons:
        button.draw(screen)

    # Draw Shrink Nodes Input Field
    shrink_label = FONT.render("Shrink Pairs:", True, WHITE)
    screen.blit(shrink_label, (WIDTH - 300, HEIGHT - PANEL_HEIGHT + 100))
    shrink_input_rect = pygame.Rect(WIDTH - 200, HEIGHT - PANEL_HEIGHT + 100, 50, 30)
    pygame.draw.rect(screen, LIGHT_BLUE, shrink_input_rect, 2)
    shrink_text = FONT.render(shrink_input_text, True, BLACK)
    screen.blit(shrink_text, (shrink_input_rect.x + 5, shrink_input_rect.y + 5))

    # Draw Energy Display
    energy_text = FONT.render(f"Energy: {energy:.2f}", True, WHITE)
    screen.blit(energy_text, (WIDTH // 2 - energy_text.get_width() // 2, HEIGHT - PANEL_HEIGHT + 150))

    # **Newly Added: Display Number of Vertices and Edges**
    vertex_count_text = FONT.render(f"Vertices: {num_vertices}", True, WHITE)
    edge_count_text = FONT.render(f"Edges: {num_edges}", True, WHITE)
    screen.blit(vertex_count_text, (WIDTH - 300, HEIGHT - PANEL_HEIGHT + 50))
    screen.blit(edge_count_text, (WIDTH - 300, HEIGHT - PANEL_HEIGHT + 80))
    # **End of Newly Added Code**

    # Instructions
    instructions = [
        "Instructions:",
        "1. Click on two connected nodes to select the initial edge.",
        "2. Adjust the slider to set the desired path length.",
        "3. The path will automatically extend up to the slider's value.",
        "4. Press 'R' to reset the traversal.",
        "5. Click 'Refine' to perform barycentric refinement.",
        "6. Enter the number of node pairs to shrink and click 'Shrink'."
    ]
    for i, text in enumerate(instructions):
        instr_surface = FONT_SMALL.render(text, True, WHITE)
        screen.blit(instr_surface, (10, HEIGHT - PANEL_HEIGHT + 10 + i * 20))

# Main Loop
def main():
    global slider_value
    clock = pygame.time.Clock()
    running = True
    dragged_node = None        # Currently dragged node

    # Zoom and Pan parameters
    zoom = 1.0
    offset_x = 0
    offset_y = 0

    # Generate the initial graph as an octahedron
    nodes, adj_matrix = generate_octahedron_graph()

    # Compute ordered neighbors for each node
    ordered_neighbors_dict = {}
    for node_index in range(len(nodes)):
        ordered = get_ordered_neighbors_by_cycle(adj_matrix, node_index)
        if ordered:
            ordered_neighbors_dict[node_index] = ordered
        else:
            # Fallback to angle-based ordering
            ordered = get_ordered_neighbors(nodes, adj_matrix, node_index)
            ordered_neighbors_dict[node_index] = ordered

    # Path Traversal Parameters
    path = []  # List to store the traversal path as node indices

    # Selection Parameters
    selected_nodes = []  # List to store selected nodes for initial edge

    # Create Slider for Path Length
    slider = Slider(
        x=WIDTH // 2 - SLIDER_WIDTH // 2,
        y=HEIGHT - PANEL_HEIGHT // 2 - SLIDER_HEIGHT // 2,
        width=SLIDER_WIDTH,
        height=SLIDER_HEIGHT,
        min_val=MIN_SLIDER,
        max_val=MAX_SLIDER,
        initial_val=slider_value,
        label="Path Length"
    )

    # Create Buttons for Barycentric Refinement and Shrink Nodes
    refine_button = Button(
        x=WIDTH - 300,
        y=HEIGHT - PANEL_HEIGHT + 140,
        width=REFINE_BUTTON_WIDTH,
        height=REFINE_BUTTON_HEIGHT,
        text="Refine"
    )

    shrink_button = Button(
        x=WIDTH - 180,
        y=HEIGHT - PANEL_HEIGHT + 140,
        width=SHRINK_BUTTON_WIDTH,
        height=SHRINK_BUTTON_HEIGHT,
        text="Shrink"
    )

    buttons = [refine_button, shrink_button]

    # Initialize list of triangles
    triangles = find_triangles(nodes, adj_matrix)

    # Input Text for Shrink Nodes
    shrink_input_text = "1"  # Default number of pairs to shrink

    while running:
        for event in pygame.event.get():
            # Handle Slider Events
            slider.handle_event(event)

            # Handle Button Events
            if refine_button.handle_event(event):
                if len(triangles) == 0:
                    print("No triangles available to refine. Please generate a graph with triangles.")
                else:
                    # Perform one refinement step
                    nodes, adj_matrix = generate_refined_graph(nodes, adj_matrix, num_refinements=1)
                    print("Performed one barycentric refinement.")

            if shrink_button.handle_event(event):
                try:
                    num_pairs = int(shrink_input_text)
                    if num_pairs <= 0:
                        raise ValueError
                    nodes, adj_matrix, pairs = collapse_nodes(nodes, adj_matrix, num_pairs)
                    # Recompute ordered neighbors after collapsing nodes
                    ordered_neighbors_dict.clear()
                    for node_index in range(len(nodes)):
                        ordered = get_ordered_neighbors_by_cycle(adj_matrix, node_index)
                        if ordered:
                            ordered_neighbors_dict[node_index] = ordered
                        else:
                            # Fallback to angle-based ordering
                            ordered = get_ordered_neighbors(nodes, adj_matrix, node_index)
                            ordered_neighbors_dict[node_index] = ordered
                    # Recompute triangles after shrinking
                    triangles = find_triangles(nodes, adj_matrix)
                    print(f"Collapsed {num_pairs} pairs of nodes.")
                except ValueError:
                    print("Invalid input for number of node pairs to shrink.")

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos

                # Check if clicking within the drawing area
                if mouse_y < HEIGHT - PANEL_HEIGHT:
                    # Check if a node is clicked
                    for idx, node in enumerate(nodes):
                        node_screen_x = (node.x + offset_x) * zoom
                        node_screen_y = (node.y + offset_y) * zoom
                        node_radius = 6
                        distance_sq = (mouse_x - node_screen_x) ** 2 + (mouse_y - node_screen_y) ** 2
                        if distance_sq <= (node_radius + 5) ** 2:  # 5 pixels margin
                            # Select node for initial edge
                            if len(selected_nodes) < 2:
                                selected_nodes.append(idx)
                                print(f"Selected node {idx} for initial edge.")
                                if len(selected_nodes) == 2:
                                    node_a, node_b = selected_nodes
                                    if adj_matrix[node_a][node_b]:
                                        path = [node_a, node_b]
                                        print(f"Initial path: {path}")
                                    else:
                                        print(f"No edge exists between node {node_a} and node {node_b}.")
                                        selected_nodes = []  # Reset selection
                            break  # Only one node can be selected at a time

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset traversal
                    path = []
                    selected_nodes = []
                    print("Traversal path reset.")

                # Handle Shrink Nodes Input
                if event.key == pygame.K_BACKSPACE:
                    shrink_input_text = shrink_input_text[:-1]
                elif event.key == pygame.K_RETURN:
                    pass  # Do nothing on Enter
                else:
                    # Limit input length and ensure it's a digit
                    if len(shrink_input_text) < 3 and event.unicode.isdigit():
                        shrink_input_text += event.unicode

        # Handle Node Dragging
        pressed = pygame.mouse.get_pressed()
        if pressed[0]:  # Left mouse button
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if mouse_y < HEIGHT - PANEL_HEIGHT:
                for node in nodes:
                    node_screen_x = (node.x + offset_x) * zoom
                    node_screen_y = (node.y + offset_y) * zoom
                    node_radius = 6
                    distance_sq = (mouse_x - node_screen_x) ** 2 + (mouse_y - node_screen_y) ** 2
                    if distance_sq <= (node_radius + 5) ** 2:
                        dragged_node = node
                        node.fixed = True
                        break
        else:
            if dragged_node:
                dragged_node.fixed = False
                dragged_node = None

        # Handle Node Movement
        if dragged_node:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            dragged_node.x = (mouse_x / zoom) - offset_x
            dragged_node.y = (mouse_y / zoom) - offset_y

        # Automatic Path Propagation Based on Slider Value
        if len(path) < slider_value and len(path) >= 2:
            current_node = path[-1]
            previous_node = path[-2]
            antipodal = find_antipodal_neighbor(adj_matrix, previous_node, current_node)
            if antipodal is not None:
                if antipodal in path:
                    print("Cycle detected. Traversal stopped.")
                else:
                    path.append(antipodal)
                    print(f"Extended path to node {antipodal}.")
            else:
                print("No antipodal neighbor found. Traversal stopped.")

        # Perform force iterations (optional: can be adjusted for performance)
        for _ in range(5):  # Fixed number of force iterations per frame for stability
            apply_forces(nodes, adj_matrix)

        # Compute energy after force application
        energy = compute_energy(nodes, adj_matrix)

        # **Newly Added: Compute Number of Vertices and Edges**
        num_vertices = len(nodes)
        num_edges = sum([len(node.edges) for node in nodes]) // 2  # Each edge is counted twice
        # **End of Newly Added Code**

        # Clear the screen and draw the graph
        draw_graph(screen, nodes, adj_matrix, zoom, offset_x, offset_y, dragged_node)

        # Highlight selected initial edge
        if len(selected_nodes) == 2:
            node_a, node_b = selected_nodes
            x1 = int((nodes[node_a].x + offset_x) * zoom)
            y1 = int((nodes[node_a].y + offset_y) * zoom)
            x2 = int((nodes[node_b].x + offset_x) * zoom)
            y2 = int((nodes[node_b].y + offset_y) * zoom)
            pygame.draw.line(screen, BLUE, (x1, y1), (x2, y2), 3)

        # Draw Traversal Path
        draw_traversal_path(screen, nodes, path, zoom, offset_x, offset_y)

        # **Newly Modified: Pass Vertex and Edge Counts to draw_ui**
        draw_ui(screen, [slider], buttons, energy, shrink_input_text, num_vertices, num_edges)
        # **End of Newly Modified Code**

        # Update the display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
