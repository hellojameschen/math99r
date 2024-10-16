import pygame
import random
import math

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Animated Force-Directed Graph with Barycentric Refinement and Speed Control")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)

# Graph settings
NUM_REFINEMENTS = 2  # Number of barycentric refinements
# Slider parameters
MIN_ITERATIONS = 1
MAX_ITERATIONS = 100  # Allows up to 100 iterations per frame
iterations_per_frame = 1  # Default iterations per frame

# Physics parameters
REPULSION_STRENGTH = 5000  # Increased strength for better separation
ATTRACTION_STRENGTH = 0.01  # Strength of the attraction along edges
DAMPING = 0.9  # Damping factor to reduce velocity over time
TIME_STEP = 0.1  # Small time step for accurate simulation

# Font setup for slider text
pygame.font.init()
FONT = pygame.font.SysFont(None, 24)

# Node class to store position, velocity, and edges
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.edges = []
        self.fx = 0.0
        self.fy = 0.0

# Generate a graph starting with a triangle and applying barycentric refinements
def generate_refined_triangle(num_refinements):
    # Initial triangle vertices
    node0 = Node(WIDTH / 2, HEIGHT / 4)
    node1 = Node(WIDTH / 4, 3 * HEIGHT / 4)
    node2 = Node(3 * WIDTH / 4, 3 * HEIGHT / 4)
    nodes = [node0, node1, node2]
    triangles = [(0, 1, 2)]

    for _ in range(num_refinements):
        new_triangles = []
        midpoint_indices = {}
        centroid_indices = {}
        for tri in triangles:
            a_idx, b_idx, c_idx = tri
            a = nodes[a_idx]
            b = nodes[b_idx]
            c = nodes[c_idx]

            # Compute midpoints of each side
            edges = [(a_idx, b_idx), (b_idx, c_idx), (c_idx, a_idx)]
            mids = []
            for edge in edges:
                edge_sorted = tuple(sorted(edge))
                if edge_sorted in midpoint_indices:
                    m_idx = midpoint_indices[edge_sorted]
                else:
                    # Create midpoint node
                    ax, ay = nodes[edge[0]].x, nodes[edge[0]].y
                    bx, by = nodes[edge[1]].x, nodes[edge[1]].y
                    mx, my = (ax + bx) / 2, (ay + by) / 2
                    m_idx = len(nodes)
                    nodes.append(Node(mx, my))
                    midpoint_indices[edge_sorted] = m_idx
                    # Add edges between midpoints and original nodes
                    nodes[edge[0]].edges.append(m_idx)
                    nodes[edge[1]].edges.append(m_idx)
                    nodes[m_idx].edges.extend([edge[0], edge[1]])
                mids.append(m_idx)

            # Compute centroid
            centroid_x = (a.x + b.x + c.x) / 3
            centroid_y = (a.y + b.y + c.y) / 3
            centroid_idx = len(nodes)
            nodes.append(Node(centroid_x, centroid_y))

            # Add edges from centroid to vertices
            nodes[centroid_idx].edges.extend([a_idx, b_idx, c_idx])
            nodes[a_idx].edges.append(centroid_idx)
            nodes[b_idx].edges.append(centroid_idx)
            nodes[c_idx].edges.append(centroid_idx)

            # Add edges from centroid to midpoints
            for m_idx in mids:
                nodes[centroid_idx].edges.append(m_idx)
                nodes[m_idx].edges.append(centroid_idx)

            # Subdivide triangle into 6 smaller triangles
            m_ab, m_bc, m_ca = mids
            new_triangles.extend([
                (a_idx, m_ab, centroid_idx),
                (b_idx, m_bc, centroid_idx),
                (c_idx, m_ca, centroid_idx),
                (m_ab, b_idx, centroid_idx),
                (m_bc, c_idx, centroid_idx),
                (m_ca, a_idx, centroid_idx)
            ])
        triangles = new_triangles

    # Ensure all edges are unique and bidirectional
    for node in nodes:
        node.edges = list(set(node.edges))

    return nodes

# Apply forces to nodes
def apply_forces(nodes):
    # Reset forces
    for node in nodes:
        node.fx = 0.0
        node.fy = 0.0

    # Repulsion between nodes
    for i, node1 in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            node2 = nodes[j]
            dx = node1.x - node2.x
            dy = node1.y - node2.y
            distance_sq = dx * dx + dy * dy
            if distance_sq == 0:
                continue  # Skip if positions are the same to avoid division by zero
            repulsion_force = REPULSION_STRENGTH / distance_sq
            distance = math.sqrt(distance_sq)
            fx = (dx / distance) * repulsion_force
            fy = (dy / distance) * repulsion_force
            node1.fx += fx
            node1.fy += fy
            node2.fx -= fx  # Newton's third law
            node2.fy -= fy

    # Attraction along edges
    for node_idx, node in enumerate(nodes):
        for edge_index in node.edges:
            if edge_index < node_idx:
                continue  # Avoid double processing
            other_node = nodes[edge_index]
            dx = other_node.x - node.x
            dy = other_node.y - node.y
            distance = math.hypot(dx, dy)
            if distance == 0:
                continue  # Skip if positions are the same
            attraction_force = ATTRACTION_STRENGTH * distance
            fx = (dx / distance) * attraction_force
            fy = (dy / distance) * attraction_force
            node.fx += fx
            node.fy += fy
            other_node.fx -= fx  # Newton's third law
            other_node.fy -= fy

    # Update positions based on forces
    for node in nodes:
        node.vx = (node.vx + node.fx * TIME_STEP) * DAMPING
        node.vy = (node.vy + node.fy * TIME_STEP) * DAMPING
        node.x += node.vx * TIME_STEP
        node.y += node.vy * TIME_STEP

        # Keep nodes within the screen boundaries
        node.x = max(10, min(WIDTH - 10, node.x))
        node.y = max(10, min(HEIGHT - 10, node.y))

# Draw the graph
def draw_graph(screen, nodes):
    screen.fill(WHITE)
    
    # Draw edges
    for node_idx, node in enumerate(nodes):
        for edge_index in node.edges:
            # To prevent drawing the same edge twice
            if edge_index < node_idx:
                continue
            other_node = nodes[edge_index]
            pygame.draw.line(screen, BLACK, (node.x, node.y), (other_node.x, other_node.y), 1)
    
    # Draw nodes
    for node in nodes:
        pygame.draw.circle(screen, RED, (int(node.x), int(node.y)), 3)
    
    # Optionally, display some information
    # Uncomment below lines to display the number of nodes and edges
    # font = pygame.font.SysFont(None, 24)
    # text = font.render(f"Nodes: {len(nodes)}", True, BLACK)
    # screen.blit(text, (10, 10))
    # pygame.display.flip()

# Draw the slider for speed control
def draw_slider(screen, iterations_per_frame):
    # Slider background
    pygame.draw.rect(screen, GRAY, (50, HEIGHT - 50, 700, 20))
    # Slider handle
    slider_position = (iterations_per_frame - MIN_ITERATIONS) / (MAX_ITERATIONS - MIN_ITERATIONS)
    handle_x = 50 + int(slider_position * 700)
    pygame.draw.circle(screen, BLUE, (handle_x, HEIGHT - 40), 10)
    
    # Draw slider label
    font = pygame.font.SysFont(None, 24)
    label = font.render("Speed Control:", True, BLACK)
    screen.blit(label, (50, HEIGHT - 80))
    
    # Draw current speed
    speed_text = font.render(f"Speed: {iterations_per_frame}x", True, BLACK)
    screen.blit(speed_text, (WIDTH // 2 - speed_text.get_width() // 2, HEIGHT - 80))

# Main loop
def main():
    global iterations_per_frame
    clock = pygame.time.Clock()
    running = True
    dragging = False  # To track if the slider is being dragged

    # Generate the graph
    nodes = generate_refined_triangle(NUM_REFINEMENTS)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if 50 <= event.pos[0] <= 750 and HEIGHT - 60 <= event.pos[1] <= HEIGHT - 20:
                    dragging = True  # Start dragging
                    # Update the iterations_per_frame based on mouse position
                    x = min(max(event.pos[0], 50), 750)
                    slider_position = (x - 50) / 700
                    iterations_per_frame = int(MIN_ITERATIONS + slider_position * (MAX_ITERATIONS - MIN_ITERATIONS))

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False  # Stop dragging

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    x = min(max(event.pos[0], 50), 750)
                    slider_position = (x - 50) / 700
                    iterations_per_frame = int(MIN_ITERATIONS + slider_position * (MAX_ITERATIONS - MIN_ITERATIONS))

        # Perform multiple iterations per frame
        for _ in range(iterations_per_frame):
            apply_forces(nodes)

        draw_graph(screen, nodes)
        draw_slider(screen, iterations_per_frame)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
