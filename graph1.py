import pygame
import random
import math

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Animated Force-Directed Graph Visualization with Speed Control")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)

# Graph settings
NUM_NODES = 10  # Number of nodes
EDGE_PROBABILITY = 0.3  # Probability of edge creation between two nodes

# Physics parameters
REPULSION_STRENGTH = 100  # Strength of the repulsion between nodes
ATTRACTION_STRENGTH = 0.001  # Strength of the attraction along edges
DAMPING = 0.85  # Damping factor to reduce velocity over time
TIME_STEP = 1.0  # Default time step

# Time step limits for the slider
MIN_TIME_STEP = 0.1
MAX_TIME_STEP = 100  # Adjusted to allow for at least 100x speed

# Precompute logarithms for efficiency
log_min = math.log10(MIN_TIME_STEP)
log_max = math.log10(MAX_TIME_STEP)
log_range = log_max - log_min

# Node class to store position, velocity, and edges
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.edges = []

# Generate random graph
def generate_random_graph(num_nodes, edge_probability):
    nodes = [Node(random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)) for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_probability:
                nodes[i].edges.append(j)
                nodes[j].edges.append(i)
    return nodes

# Apply forces to nodes
def apply_forces(nodes, time_step):
    # Repulsion between all pairs of nodes
    for i, node1 in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            node2 = nodes[j]
            dx = node1.x - node2.x
            dy = node1.y - node2.y
            distance = math.hypot(dx, dy)
            if distance < 1:
                distance = 1  # Avoid division by zero
            repulsion_force = (REPULSION_STRENGTH / (distance ** 2)) * time_step
            force_x = (dx / distance) * repulsion_force
            force_y = (dy / distance) * repulsion_force
            node1.vx += force_x
            node1.vy += force_y
            node2.vx -= force_x  # Newton's third law
            node2.vy -= force_y
    
    # Attraction along edges
    for node in nodes:
        for edge_index in node.edges:
            other_node = nodes[edge_index]
            dx = other_node.x - node.x
            dy = other_node.y - node.y
            distance = math.hypot(dx, dy)
            if distance < 1:
                distance = 1  # Avoid division by zero
            attraction_force = (ATTRACTION_STRENGTH * distance) * time_step
            force_x = (dx / distance) * attraction_force
            force_y = (dy / distance) * attraction_force
            node.vx += force_x
            node.vy += force_y
            other_node.vx -= force_x  # Newton's third law
            other_node.vy -= force_y

    # Update positions based on velocity and apply damping
    for node in nodes:
        node.vx *= DAMPING
        node.vy *= DAMPING
        node.x += node.vx
        node.y += node.vy

        # Keep nodes within the screen boundaries
        node.x = max(10, min(WIDTH - 10, node.x))
        node.y = max(10, min(HEIGHT - 10, node.y))

# Draw the graph
def draw_graph(screen, nodes):
    screen.fill(WHITE)
    
    # Draw edges
    for node in nodes:
        for edge_index in node.edges:
            other_node = nodes[edge_index]
            pygame.draw.line(screen, BLACK, (node.x, node.y), (other_node.x, other_node.y), 2)
    
    # Draw nodes
    for node in nodes:
        pygame.draw.circle(screen, RED, (int(node.x), int(node.y)), 10)

# Draw the slider for speed control
def draw_slider(screen, time_step):
    # Slider background
    pygame.draw.rect(screen, GRAY, (50, HEIGHT - 50, 700, 20))
    # Slider handle
    if time_step <= 0:
        time_step = MIN_TIME_STEP
    log_time_step = math.log10(time_step)
    slider_position = (log_time_step - log_min) / log_range
    handle_x = 50 + int(slider_position * 700)
    pygame.draw.circle(screen, BLUE, (handle_x, HEIGHT - 40), 10)

# Main loop
def main():
    global TIME_STEP
    clock = pygame.time.Clock()
    running = True
    dragging = False  # To track if the slider is being dragged
    
    # Generate the graph
    nodes = generate_random_graph(NUM_NODES, EDGE_PROBABILITY)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if 50 <= event.pos[0] <= 750 and HEIGHT - 60 <= event.pos[1] <= HEIGHT - 20:
                    dragging = True  # Start dragging
                    slider_position = (event.pos[0] - 50) / 700
                    log_time_step = log_min + slider_position * log_range
                    TIME_STEP = 10 ** log_time_step

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False  # Stop dragging

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    x = min(max(event.pos[0], 50), 750)
                    slider_position = (x - 50) / 700
                    log_time_step = log_min + slider_position * log_range
                    TIME_STEP = 10 ** log_time_step
        
        apply_forces(nodes, TIME_STEP)  # Update node positions based on forces
        draw_graph(screen, nodes)
        draw_slider(screen, TIME_STEP)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
