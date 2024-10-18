import pygame
import math
import random

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
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 149, 237)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Slider Parameters
MIN_ITERATIONS = 1
MAX_ITERATIONS = 100
iterations_per_frame = 1

MIN_ZOOM = 0.1
MAX_ZOOM = 2.0
zoom_level = 1.0

# Physics Parameters
REPULSION_STRENGTH = 5000
ATTRACTION_STRENGTH = 0.01
DAMPING = 0.9
TIME_STEP = 0.1

# Fonts
FONT = pygame.font.SysFont(None, 24)

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

# Generate Octahedron Graph
def generate_octahedron_graph():
    nodes = []
    adj_matrix = [[0]*6 for _ in range(6)]  # 6 nodes for an octahedron

    # Define positions for an octahedron centered on the screen
    center_x, center_y = WIDTH // 2, (HEIGHT - PANEL_HEIGHT) // 2
    radius = 150  # Distance from center to each vertex

    # Octahedron has 6 vertices: one top, one bottom, and four around the center
    positions = [
        (center_x, center_y - radius),  # Top node
        (center_x, center_y + radius),  # Bottom node
        (center_x - radius, center_y),  # Left node
        (center_x + radius, center_y),  # Right node
        (center_x, center_y - radius // 2),  # Upper-middle node
        (center_x, center_y + radius // 2)   # Lower-middle node
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

# Main Loop
def main():
    global iterations_per_frame, zoom_level
    clock = pygame.time.Clock()
    running = True
    dragging_speed = False     # To track if the speed slider is being dragged
    dragging_zoom = False      # To track if the zoom slider is being dragged
    panning = False            # To track if panning is active
    pan_start_pos = (0, 0)     # Starting position for panning
    dragged_node = None        # Currently dragged node

    # Zoom and Pan parameters
    zoom = zoom_level
    offset_x = 0
    offset_y = 0

    # Generate the initial graph as an octahedron
    nodes, adj_matrix = generate_octahedron_graph()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos

                # Check if drag mode is active and handle dragging
                for idx, node in enumerate(nodes):
                    # Transform node position to screen coordinates
                    node_screen_x = (node.x + offset_x) * zoom
                    node_screen_y = (node.y + offset_y) * zoom
                    node_radius = 4
                    distance_sq = (mouse_x - node_screen_x) ** 2 + (mouse_y - node_screen_y) ** 2
                    if distance_sq <= (node_radius + 5) ** 2:  # 5 pixels margin
                        # Initiate dragging the node
                        dragged_node = node
                        node.fixed = True
                        break  # Only one node can be interacted with at a time

            elif event.type == pygame.MOUSEBUTTONUP:
                if dragged_node:
                    dragged_node.fixed = False
                    dragged_node = None

            elif event.type == pygame.MOUSEMOTION:
                # Update position of the dragged node
                if dragged_node:
                    # Convert mouse position to world coordinates
                    world_x = (event.pos[0] / zoom) - offset_x
                    world_y = (event.pos[1] / zoom) - offset_y
                    dragged_node.x = world_x
                    dragged_node.y = world_y

        # Perform multiple iterations per frame
        for _ in range(iterations_per_frame):
            apply_forces(nodes, adj_matrix)

        # Compute energy after force application
        energy = compute_energy(nodes, adj_matrix)

        # Clear the screen and draw the graph
        draw_graph(screen, nodes, adj_matrix, zoom, offset_x, offset_y, dragged_node)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
