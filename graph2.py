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
NUM_NODES = 20  # Increased number of nodes for better visualization
EDGE_PROBABILITY = 0.5  # Probability of edge creation between two nodes

# Physics parameters
REPULSION_STRENGTH = 10000  # Strength of the repulsion between nodes
ATTRACTION_STRENGTH = 0.01  # Strength of the attraction along edges
DAMPING = 0.9  # Damping factor to reduce velocity over time
TIME_STEP = 0.1  # Small time step for accurate simulation

# Simulation parameters
MIN_ITERATIONS = 1
MAX_ITERATIONS = 100  # Allows up to 100 iterations per frame
iterations_per_frame = 1  # Default iterations per frame

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
def apply_forces(nodes):
    # Reset forces
    for node in nodes:
        node.fx = 0
        node.fy = 0

    # Repulsion between nodes
    for i, node1 in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            node2 = nodes[j]
            dx = node1.x - node2.x
            dy = node1.y - node2.y
            distance_sq = dx * dx + dy * dy
            if distance_sq == 0:
                distance_sq = 0.01  # Avoid division by zero
            distance = math.sqrt(distance_sq)
            repulsion_force = REPULSION_STRENGTH / distance_sq
            fx = (dx / distance) * repulsion_force
            fy = (dy / distance) * repulsion_force
            node1.fx += fx
            node1.fy += fy
            node2.fx -= fx  # Newton's third law
            node2.fy -= fy

    # Attraction along edges
    for node in nodes:
        for edge_index in node.edges:
            other_node = nodes[edge_index]
            dx = other_node.x - node.x
            dy = other_node.y - node.y
            distance = math.hypot(dx, dy)
            if distance == 0:
                distance = 0.01  # Avoid division by zero
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
    for node in nodes:
        for edge_index in node.edges:
            other_node = nodes[edge_index]
            pygame.draw.line(screen, BLACK, (node.x, node.y), (other_node.x, other_node.y), 2)
    
    # Draw nodes
    for node in nodes:
        pygame.draw.circle(screen, RED, (int(node.x), int(node.y)), 10)

# Draw the slider for speed control
def draw_slider(screen, iterations_per_frame):
    # Slider background
    pygame.draw.rect(screen, GRAY, (50, HEIGHT - 50, 700, 20))
    # Slider handle
    slider_position = (iterations_per_frame - MIN_ITERATIONS) / (MAX_ITERATIONS - MIN_ITERATIONS)
    handle_x = 50 + int(slider_position * 700)
    pygame.draw.circle(screen, BLUE, (handle_x, HEIGHT - 40), 10)
    # Draw text
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Speed: {iterations_per_frame}x", True, BLACK)
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT - 80))

# Main loop
def main():
    global iterations_per_frame
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
