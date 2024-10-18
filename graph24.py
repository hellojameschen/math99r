import pygame
import math
import sys

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 750  # Screen dimensions
PANEL_HEIGHT = 200        # Height reserved for the UI panel
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Antipodal Traversal with Force-Directed Graph")

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)  # Dark gray for unreachable nodes
LIGHT_BLUE = (173, 216, 230)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)      # Color for highlighted edges
PURPLE = (128, 0, 128)      # Distinct color for the source node
BUTTON_COLOR = (70, 130, 180)        # Steel Blue for buttons
BUTTON_HOVER_COLOR = (100, 149, 237) # Cornflower Blue for button hover
BLUE = (0, 0, 255)          # General purpose blue
RED = (255, 0, 0)           # Color for nodes at highlight distance

# Slider Parameters
MIN_SLIDER = 1
MAX_SLIDER = 20
SLIDER_WIDTH = 200
SLIDER_HEIGHT = 20
slider_value = 5  # Initial path length

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
        self.fixed = False  # Indicates if the node is being dragged

        # Attributes for Antipodal Traversal
        self.visited = False
        self.in_path = False

        # Edges list to keep track of connections
        self.edges = []

# Slider Class
class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val):
        self.rect = pygame.Rect(x, y, width, height)
        self.min = min_val
        self.max = max_val
        self.value = initial_val
        self.handle_radius = height // 2
        self.handle_x = self.get_handle_x()
        self.dragging = False  # Initialize dragging state

    def get_handle_x(self):
        proportion = (self.value - self.min) / (self.max - self.min)
        return self.rect.x + int(proportion * self.rect.width)

    def draw(self, surface):
        # Draw slider line
        pygame.draw.line(surface, BLACK, (self.rect.x, self.rect.centery),
                         (self.rect.x + self.rect.width, self.rect.centery), 2)
        # Draw handle
        pygame.draw.circle(surface, BLUE, (self.handle_x, self.rect.centery), self.handle_radius)

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

# Antipodal Traversal Algorithm
def antipodal_traversal(nodes, adj_matrix, start_idx, path_length):
    path = [start_idx]
    current_node = start_idx
    previous_node = None

    for _ in range(path_length - 1):
        neighbors = nodes[current_node].edges
        if not neighbors:
            print("No more neighbors to traverse.")
            break

        if previous_node is None:
            # If at the start, choose the first neighbor
            next_node = neighbors[0]
        else:
            # Select the antipodal neighbor relative to the previous node
            ordered_neighbors = get_ordered_neighbors_by_cycle(adj_matrix, current_node)
            if not ordered_neighbors:
                ordered_neighbors = get_ordered_neighbors_by_angle(nodes, adj_matrix, current_node)
            if not ordered_neighbors:
                print(f"No ordered neighbors found for node {current_node}.")
                break

            try:
                idx = ordered_neighbors.index(previous_node)
                antipodal_idx = (idx + len(ordered_neighbors) // 2) % len(ordered_neighbors)
                next_node = ordered_neighbors[antipodal_idx]
            except ValueError:
                # Previous node not in ordered neighbors
                next_node = ordered_neighbors[0]  # Fallback to first neighbor

        if next_node in path:
            print("Cycle detected. Stopping traversal.")
            break

        path.append(next_node)
        previous_node, current_node = current_node, next_node

    return path

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
def get_ordered_neighbors_by_angle(nodes, adj_matrix, node_index):
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

# Draw the Graph
def draw_graph(screen, nodes, adj_matrix, zoom, offset_x, offset_y, dragged_node, path):
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

    # Highlight traversal path
    for i in range(len(path) - 1):
        node_a = nodes[path[i]]
        node_b = nodes[path[i + 1]]
        x1 = int((node_a.x + offset_x) * zoom)
        y1 = int((node_a.y + offset_y) * zoom)
        x2 = int((node_b.x + offset_x) * zoom)
        y2 = int((node_b.y + offset_y) * zoom)
        pygame.draw.line(screen, RED, (x1, y1), (x2, y2), 2)

    # Draw nodes
    for idx, node in enumerate(nodes):
        # Apply zoom and offset
        x = (node.x + offset_x) * zoom
        y = (node.y + offset_y) * zoom
        # Only draw nodes that are within the visible screen area
        if 0 <= x <= WIDTH and 0 <= y <= (HEIGHT - PANEL_HEIGHT):
            if idx == path[0]:
                color = PURPLE  # Source node
            elif node.in_path:
                color = GREEN  # Nodes in the traversal path
            else:
                color = GRAY  # Other nodes

            # Override color if the node is being dragged
            if node == dragged_node:
                color = YELLOW

            pygame.draw.circle(screen, color, (int(x), int(y)), 6 if node == dragged_node else 4)

            # Draw a circle around the source node
            if idx == path[0]:
                pygame.draw.circle(screen, WHITE, (int(x), int(y)), 10, 2)  # White circle with radius 10 and thickness 2

# Draw the UI Panel
def draw_ui(screen, slider, path_length, energy, instructions):
    # Define the panel area
    panel_rect = pygame.Rect(0, HEIGHT - PANEL_HEIGHT, WIDTH, PANEL_HEIGHT)
    pygame.draw.rect(screen, DARK_GRAY, panel_rect)

    # Draw Slider and its label
    slider.draw(screen)
    slider_label = FONT.render(f"Antipodal Path Length: {path_length}", True, WHITE)
    screen.blit(slider_label, (slider.rect.x, slider.rect.y - 30))

    # Display Energy
    energy_text = FONT.render(f"Energy: {energy:.2f}", True, WHITE)
    screen.blit(energy_text, (WIDTH - 200, HEIGHT - PANEL_HEIGHT + 20))

    # Instructions
    for idx, text in enumerate(instructions):
        instr_surface = FONT_SMALL.render(text, True, WHITE)
        screen.blit(instr_surface, (10, HEIGHT - PANEL_HEIGHT + 60 + idx * 20))

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

    # Path Traversal Parameters
    path = []  # List to store the traversal path as node indices

    # Create Slider
    slider = Slider(
        x=WIDTH // 2 - SLIDER_WIDTH // 2,
        y=HEIGHT - PANEL_HEIGHT // 2 - SLIDER_HEIGHT // 2,
        width=SLIDER_WIDTH,
        height=SLIDER_HEIGHT,
        min_val=MIN_SLIDER,
        max_val=MAX_SLIDER,
        initial_val=slider_value
    )

    # Instructions
    instructions = [
        "Instructions:",
        "1. Click on a node to start antipodal traversal.",
        "2. Adjust the slider to set the desired path length.",
        "3. Drag nodes to reposition them.",
    ]

    while running:
        for event in pygame.event.get():
            # Handle Slider Events
            slider.handle_event(event)

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
                            # Start traversal from this node
                            path = antipodal_traversal(nodes, adj_matrix, idx, slider_value)
                            # Mark nodes in path
                            for node_idx in path:
                                nodes[node_idx].in_path = True
                            print(f"Traversal path: {path}")
                            break  # Only one node can be selected at a time

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset traversal
                    path = []
                    for node in nodes:
                        node.in_path = False
                    print("Traversal path reset.")

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
        if len(path) < slider_value and len(path) >= 1:
            current_node = path[-1]
            previous_node = path[-2] if len(path) >= 2 else None
            antipodal = None
            if previous_node is not None:
                # Select the antipodal neighbor relative to the previous node
                ordered_neighbors = get_ordered_neighbors_by_cycle(adj_matrix, current_node)
                if not ordered_neighbors:
                    ordered_neighbors = get_ordered_neighbors_by_angle(nodes, adj_matrix, current_node)
                if not ordered_neighbors:
                    print(f"No ordered neighbors found for node {current_node}.")
                else:
                    try:
                        idx = ordered_neighbors.index(previous_node)
                        antipodal_idx = (idx + len(ordered_neighbors) // 2) % len(ordered_neighbors)
                        antipodal = ordered_neighbors[antipodal_idx]
                    except ValueError:
                        # Previous node not in ordered neighbors
                        antipodal = ordered_neighbors[0]  # Fallback to first neighbor
            else:
                # If no previous node, choose the first neighbor
                neighbors = nodes[current_node].edges
                if neighbors:
                    antipodal = neighbors[0]

            if antipodal is not None:
                if antipodal in path:
                    print("Cycle detected. Traversal stopped.")
                elif len(path) < slider_value:
                    path.append(antipodal)
                    nodes[antipodal].in_path = True
                    print(f"Extended path to node {antipodal}.")
            else:
                print("No antipodal neighbor found. Traversal stopped.")

        # Perform force iterations
        for _ in range(slider_value):
            apply_forces(nodes, adj_matrix)

        # Compute energy after force application
        energy = compute_energy(nodes, adj_matrix)

        # Clear the screen and draw the graph and UI elements
        draw_graph(screen, nodes, adj_matrix, zoom, offset_x, offset_y, dragged_node, path)
        draw_ui(screen, slider, slider_value, energy, instructions)

        # Update the display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
