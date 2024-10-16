import pygame
import math
import random
import numpy as np  # For matrix operations

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 750  # Screen dimensions
PANEL_HEIGHT = 200        # Height reserved for the UI panel
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Animated Force-Directed Graph with Markov Wavefront")

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
MIN_ITERATIONS = 1
MAX_ITERATIONS = 100
iterations_per_frame = 1

MIN_ZOOM = 0.1
MAX_ZOOM = 2.0
zoom_level = 1.0

MIN_WAVE_STEP = 0
MAX_WAVE_STEP = 20  # Adjust based on expected graph size
wave_step = 0  # Default wave step

# Physics Parameters
REPULSION_STRENGTH = 5000
ATTRACTION_STRENGTH = 0.01
DAMPING = 0.9
TIME_STEP = 0.1

# Fonts
FONT = pygame.font.SysFont(None, 24)
FONT_SMALL = pygame.font.SysFont(None, 20)

# Node Class
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.fx = 0.0
        self.fy = 0.0
        self.fixed = False  # Indicates if the node is being dragged
        self.color = GRAY   # Default color

# Generate Random Graph
def generate_random_graph(num_nodes, edge_probability):
    nodes = []
    adj_matrix = [[0]*num_nodes for _ in range(num_nodes)]
    margin = 50  # Margin to prevent nodes from being too close to the edges
    for _ in range(num_nodes):
        x = random.randint(margin, WIDTH - margin)
        y = random.randint(margin, HEIGHT - PANEL_HEIGHT - margin)
        nodes.append(Node(x, y))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_probability:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1  # For undirected graph

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
        for j in range(num_nodes):
            if adj_matrix[i][j]:
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
        for j in range(num_nodes):
            if adj_matrix[i][j]:
                other_node = nodes[j]
                dx = other_node.x - node.x
                dy = other_node.y - node.y
                distance = math.hypot(dx, dy)
                if distance == 0:
                    continue  # Prevent division by zero
                attraction_energy = ATTRACTION_STRENGTH * distance * distance / 2
                total_energy += attraction_energy

    return total_energy

# Function to map probability to color (Blue to Red)
def probability_to_color(prob):
    """
    Maps a probability value between 0 and 1 to a color gradient from Blue (low) to Red (high).
    """
    # Clamp probability to [0,1]
    prob = max(0.0, min(1.0, prob))
    # Calculate red and blue components
    red = int(prob * 255)
    blue = int((1 - prob) * 255)
    green = 0  # Keep green at 0 for a clear gradient
    return (red, green, blue)

# Draw the Graph
def draw_graph(screen, nodes, adj_matrix, zoom, offset_x, offset_y, dragged_node, source_node, wave_step, wave_vector):
    # Define the drawing area (excluding the panel)
    drawing_area_rect = pygame.Rect(0, 0, WIDTH, HEIGHT - PANEL_HEIGHT)
    pygame.draw.rect(screen, WHITE, drawing_area_rect)

    num_nodes = len(nodes)

    # Draw edges
    for i in range(num_nodes):
        node = nodes[i]
        for j in range(i + 1, num_nodes):
            if adj_matrix[i][j]:
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
            if node == source_node:
                color = PURPLE  # Distinct color for the source node
            elif wave_vector is not None and wave_vector[idx] > 0:
                # Color based on probability, higher probability is more red
                color = probability_to_color(wave_vector[idx])
            else:
                color = GRAY  # Default color

            # Override color if the node is being dragged
            if node == dragged_node:
                color = YELLOW

            pygame.draw.circle(screen, color, (int(x), int(y)), 6 if node == dragged_node else 4)
            
            # Draw a circle around the source node
            if node == source_node:
                pygame.draw.circle(screen, WHITE, (int(x), int(y)), 10, 2)  # White circle with radius 10 and thickness 2

# Draw the Probability Legend
def draw_legend(screen):
    """
    Draws a color gradient legend indicating the mapping from probability to color.
    """
    legend_width = 200
    legend_height = 20
    legend_x = 50
    legend_y = HEIGHT - PANEL_HEIGHT + 20  # Positioned near the top of the panel

    # Create a surface for the legend
    legend_surface = pygame.Surface((legend_width, legend_height))
    for x in range(legend_width):
        # Calculate probability
        prob = x / (legend_width - 1)
        color = probability_to_color(prob)
        pygame.draw.line(legend_surface, color, (x, 0), (x, legend_height))
    
    # Draw border around the legend
    pygame.draw.rect(screen, BLACK, (legend_x - 2, legend_y - 2, legend_width + 4, legend_height + 4), 2)
    
    # Blit the legend onto the screen
    screen.blit(legend_surface, (legend_x, legend_y))
    
    # Add labels
    min_label = FONT.render("0.0", True, BLACK)
    max_label = FONT.render("1.0", True, BLACK)
    screen.blit(min_label, (legend_x, legend_y + legend_height + 5))
    screen.blit(max_label, (legend_x + legend_width - max_label.get_width(), legend_y + legend_height + 5))

# Draw the UI Panel
def draw_ui(screen, iterations_per_frame, zoom_level, num_nodes, edge_prob, energy, input_active, input_boxes, 
            refresh_button_hover, drag_mode, drag_mode_hover, wave_step, wave_slider_hover):
    # Define the panel area
    panel_rect = pygame.Rect(0, HEIGHT - PANEL_HEIGHT, WIDTH, PANEL_HEIGHT)
    pygame.draw.rect(screen, DARK_GRAY, panel_rect)

    # Slider 1: Speed Control
    speed_slider_x = 50
    speed_slider_y = HEIGHT - PANEL_HEIGHT + 50
    speed_slider_width = 200
    speed_slider_height = 20

    # Slider background
    pygame.draw.rect(screen, GRAY, (speed_slider_x, speed_slider_y, speed_slider_width, speed_slider_height))

    # Slider handle
    speed_slider_position = (iterations_per_frame - MIN_ITERATIONS) / (MAX_ITERATIONS - MIN_ITERATIONS)
    speed_handle_x = speed_slider_x + int(speed_slider_position * speed_slider_width)
    speed_handle_y = speed_slider_y + speed_slider_height // 2
    pygame.draw.circle(screen, BLUE, (speed_handle_x, speed_handle_y), 10)

    # Slider 2: Zoom Control
    zoom_slider_x = 300
    zoom_slider_y = HEIGHT - PANEL_HEIGHT + 50
    zoom_slider_width = 200
    zoom_slider_height = 20

    # Slider background
    pygame.draw.rect(screen, GRAY, (zoom_slider_x, zoom_slider_y, zoom_slider_width, zoom_slider_height))

    # Slider handle
    zoom_slider_position = (zoom_level - MIN_ZOOM) / (MAX_ZOOM - MIN_ZOOM)
    zoom_handle_x = zoom_slider_x + int(zoom_slider_position * zoom_slider_width)
    zoom_handle_y = zoom_slider_y + zoom_slider_height // 2
    pygame.draw.circle(screen, BLUE, (zoom_handle_x, zoom_handle_y), 10)

    # Slider 3: Wave Step Control
    wave_slider_x = 550
    wave_slider_y = HEIGHT - PANEL_HEIGHT + 50
    wave_slider_width = 200
    wave_slider_height = 20

    # Slider background
    pygame.draw.rect(screen, GRAY, (wave_slider_x, wave_slider_y, wave_slider_width, wave_slider_height))

    # Slider handle
    wave_slider_position = (wave_step - MIN_WAVE_STEP) / (MAX_WAVE_STEP - MIN_WAVE_STEP)
    wave_handle_x = wave_slider_x + int(wave_slider_position * wave_slider_width)
    wave_handle_y = wave_slider_y + wave_slider_height // 2
    pygame.draw.circle(screen, BLUE, (wave_handle_x, wave_handle_y), 10)

    # Draw slider labels
    label_speed = FONT.render("Speed Control:", True, WHITE)
    screen.blit(label_speed, (speed_slider_x, speed_slider_y - 30))

    label_zoom = FONT.render("Zoom Control:", True, WHITE)
    screen.blit(label_zoom, (zoom_slider_x, zoom_slider_y - 30))

    label_wave = FONT.render("Wave Step:", True, WHITE)
    screen.blit(label_wave, (wave_slider_x, wave_slider_y - 30))

    # Draw current slider values
    speed_text = FONT.render(f"Speed: {iterations_per_frame}x", True, WHITE)
    screen.blit(speed_text, (speed_slider_x + speed_slider_width // 2 - speed_text.get_width() // 2, speed_slider_y - 30))

    zoom_display = f"{zoom_level:.2f}x"
    zoom_text = FONT.render(f"Zoom: {zoom_display}", True, WHITE)
    screen.blit(zoom_text, (zoom_slider_x + zoom_slider_width // 2 - zoom_text.get_width() // 2, zoom_slider_y - 30))

    wave_text = FONT.render(f"Step: {wave_step}", True, WHITE)
    screen.blit(wave_text, (wave_slider_x + wave_slider_width // 2 - wave_text.get_width() // 2, wave_slider_y - 30))

    # Input Fields: Number of Nodes and Edge Probability
    # Number of Nodes
    nodes_label = FONT.render("Number of Nodes:", True, WHITE)
    screen.blit(nodes_label, (50, HEIGHT - PANEL_HEIGHT + 100))
    pygame.draw.rect(screen, LIGHT_BLUE if input_active['num_nodes'] else GRAY, input_boxes['num_nodes']['rect'], 2)
    nodes_text = FONT.render(input_boxes['num_nodes']['text'], True, BLACK)
    screen.blit(nodes_text, (input_boxes['num_nodes']['rect'].x + 5, input_boxes['num_nodes']['rect'].y + 5))

    # Edge Probability
    edge_label = FONT.render("Edge Probability (0-1):", True, WHITE)
    screen.blit(edge_label, (300, HEIGHT - PANEL_HEIGHT + 100))
    pygame.draw.rect(screen, LIGHT_BLUE if input_active['edge_prob'] else GRAY, input_boxes['edge_prob']['rect'], 2)
    edge_text = FONT.render(input_boxes['edge_prob']['text'], True, BLACK)
    screen.blit(edge_text, (input_boxes['edge_prob']['rect'].x + 5, input_boxes['edge_prob']['rect'].y + 5))

    # Refresh Button
    refresh_button_rect = input_boxes['refresh_button']['rect']
    pygame.draw.rect(screen, GREEN if refresh_button_hover else GRAY, refresh_button_rect)
    refresh_text = FONT.render("Refresh", True, BLACK)
    screen.blit(refresh_text, (
        refresh_button_rect.x + (refresh_button_rect.width - refresh_text.get_width()) // 2,
        refresh_button_rect.y + (refresh_button_rect.height - refresh_text.get_height()) // 2
    ))

    # Drag Mode Toggle Button
    drag_mode_button_rect = input_boxes['drag_mode_button']['rect']
    button_color = BUTTON_HOVER_COLOR if drag_mode_hover else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, drag_mode_button_rect)
    drag_mode_text = FONT.render(f"Drag Mode: {'ON' if drag_mode else 'OFF'}", True, WHITE)
    screen.blit(drag_mode_text, (
        drag_mode_button_rect.x + (drag_mode_button_rect.width - drag_mode_text.get_width()) // 2,
        drag_mode_button_rect.y + (drag_mode_button_rect.height - drag_mode_text.get_height()) // 2
    ))

    # Display Energy
    energy_text = FONT.render(f"Energy: {energy:.2f}", True, WHITE)
    screen.blit(energy_text, (350, HEIGHT - PANEL_HEIGHT + 150))

    # Draw Probability Legend
    draw_legend(screen)

    # Instructions
    instructions = [
        "Select a source node by clicking on it.",
        "Use the 'Wave Step' slider to propagate the wave.",
        "Toggle Drag Mode to drag nodes.",
        "Adjust Speed and Zoom as needed."
    ]
    for idx, text in enumerate(instructions):
        instr_text = FONT_SMALL.render(text, True, WHITE)
        screen.blit(instr_text, (50, HEIGHT - PANEL_HEIGHT + 180 + idx * 20))

# Main Loop
def main():
    global iterations_per_frame, zoom_level, wave_step
    clock = pygame.time.Clock()
    running = True
    dragging_speed = False     # To track if the speed slider is being dragged
    dragging_zoom = False      # To track if the zoom slider is being dragged
    dragging_wave = False      # To track if the wave slider is being dragged
    panning = False            # To track if panning is active
    pan_start_pos = (0, 0)     # Starting position for panning
    dragged_node = None        # Currently dragged node

    # Zoom and Pan parameters
    zoom = zoom_level
    offset_x = 0
    offset_y = 0

    # Wave propagation state
    source_node = None
    wave_vector = None  # Result of Markov matrix raised to wave_step

    # Generate the initial graph as a random graph
    nodes, adj_matrix = generate_random_graph(50, 0.05)  # Default: 50 nodes, 5% edge probability

    # Convert adjacency matrix to numpy array for Markov operations
    adj_matrix_np = np.array(adj_matrix, dtype=float)
    row_sums = adj_matrix_np.sum(axis=1)
    # Handle dangling nodes (nodes with no outgoing edges)
    row_sums[row_sums == 0] = 1
    markov_matrix = adj_matrix_np / row_sums[:, np.newaxis]

    # UI State
    input_active = {'num_nodes': False, 'edge_prob': False}
    input_boxes = {
        'num_nodes': {'rect': pygame.Rect(200, HEIGHT - PANEL_HEIGHT + 100, 80, 30), 'text': '50'},
        'edge_prob': {'rect': pygame.Rect(500, HEIGHT - PANEL_HEIGHT + 100, 80, 30), 'text': '0.05'},
        'refresh_button': {'rect': pygame.Rect(350, HEIGHT - PANEL_HEIGHT + 100, 80, 30)},
        'drag_mode_button': {'rect': pygame.Rect(350, HEIGHT - PANEL_HEIGHT + 150, 150, 30)}  # Drag Mode Button
    }
    refresh_button_hover = False
    drag_mode_button_hover = False  # Hover state for Drag Mode Button

    # Drag Mode State
    drag_mode = False  # Initially, drag mode is off

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos

                # Check if the click is within the speed slider area
                speed_slider_x = 50
                speed_slider_y = HEIGHT - PANEL_HEIGHT + 50
                speed_slider_width = 200
                speed_slider_height = 20
                handle_radius = 10
                if (speed_slider_x <= mouse_x <= speed_slider_x + speed_slider_width) and \
                   (speed_slider_y - handle_radius <= mouse_y <= speed_slider_y + speed_slider_height + handle_radius):
                    dragging_speed = True
                    # Update the iterations_per_frame based on mouse position
                    x = min(max(mouse_x, speed_slider_x), speed_slider_x + speed_slider_width)
                    slider_position = (x - speed_slider_x) / speed_slider_width
                    iterations_per_frame = max(MIN_ITERATIONS, min(int(MIN_ITERATIONS + slider_position * (MAX_ITERATIONS - MIN_ITERATIONS)), MAX_ITERATIONS))

                # Check if the click is within the zoom slider area
                zoom_slider_x = 300
                zoom_slider_y = HEIGHT - PANEL_HEIGHT + 50
                zoom_slider_width = 200
                zoom_slider_height = 20
                if (zoom_slider_x <= mouse_x <= zoom_slider_x + zoom_slider_width) and \
                   (zoom_slider_y - handle_radius <= mouse_y <= zoom_slider_y + zoom_slider_height + handle_radius):
                    dragging_zoom = True
                    # Calculate zoom level based on mouse position
                    x = min(max(mouse_x, zoom_slider_x), zoom_slider_x + zoom_slider_width)
                    slider_position = (x - zoom_slider_x) / zoom_slider_width
                    new_zoom_level = MIN_ZOOM + slider_position * (MAX_ZOOM - MIN_ZOOM)
                    
                    # Center the zoom on the screen center
                    cx, cy = WIDTH / 2, (HEIGHT - PANEL_HEIGHT) / 2
                    # Calculate world coordinates before zoom
                    wx = (cx / zoom) - offset_x
                    wy = (cy / zoom) - offset_y
                    # Update zoom level
                    zoom_level = new_zoom_level
                    zoom = zoom_level
                    # Calculate new offsets to keep (wx, wy) at the center
                    offset_x = (cx / zoom) - wx
                    offset_y = (cy / zoom) - wy

                # Check if the click is within the wave slider area
                wave_slider_x = 550
                wave_slider_y = HEIGHT - PANEL_HEIGHT + 50
                wave_slider_width = 200
                wave_slider_height = 20
                if (wave_slider_x <= mouse_x <= wave_slider_x + wave_slider_width) and \
                   (wave_slider_y - handle_radius <= mouse_y <= wave_slider_y + wave_slider_height + handle_radius):
                    dragging_wave = True
                    # Update the wave_step based on mouse position
                    x = min(max(mouse_x, wave_slider_x), wave_slider_x + wave_slider_width)
                    slider_position = (x - wave_slider_x) / wave_slider_width
                    wave_step = max(MIN_WAVE_STEP, min(int(MIN_WAVE_STEP + slider_position * (MAX_WAVE_STEP - MIN_WAVE_STEP)), MAX_WAVE_STEP))

                # Check if click is within input boxes
                if input_boxes['num_nodes']['rect'].collidepoint(event.pos):
                    input_active['num_nodes'] = True
                    input_active['edge_prob'] = False
                elif input_boxes['edge_prob']['rect'].collidepoint(event.pos):
                    input_active['edge_prob'] = True
                    input_active['num_nodes'] = False
                else:
                    input_active['num_nodes'] = False
                    input_active['edge_prob'] = False

                # Check if click is on the refresh button
                if input_boxes['refresh_button']['rect'].collidepoint(event.pos):
                    try:
                        num_nodes = int(input_boxes['num_nodes']['text'])
                        edge_prob = float(input_boxes['edge_prob']['text'])
                        if num_nodes <= 0 or not (0 <= edge_prob <= 1):
                            raise ValueError
                        nodes, adj_matrix = generate_random_graph(num_nodes, edge_prob)
                        # Recreate the Markov matrix
                        adj_matrix_np = np.array(adj_matrix, dtype=float)
                        row_sums = adj_matrix_np.sum(axis=1)
                        row_sums[row_sums == 0] = 1
                        markov_matrix = adj_matrix_np / row_sums[:, np.newaxis]
                        # Reset wave state
                        wave_step = 0
                        wave_vector = None
                        source_node = None
                        # Reset node colors
                        for node in nodes:
                            node.color = GRAY
                    except ValueError:
                        print("Invalid input for number of nodes or edge probability.")

                # Check if click is on the Drag Mode button
                if input_boxes['drag_mode_button']['rect'].collidepoint(event.pos):
                    drag_mode = not drag_mode  # Toggle drag mode
                else:
                    # If drag mode is active, handle dragging
                    if drag_mode:
                        for idx, node in enumerate(nodes):
                            # Transform node position to screen coordinates
                            node_screen_x = (node.x + offset_x) * zoom
                            node_screen_y = (node.y + offset_y) * zoom
                            # Define node radius for clicking
                            node_radius = 6
                            distance_sq = (mouse_x - node_screen_x) ** 2 + (mouse_y - node_screen_y) ** 2
                            if distance_sq <= (node_radius + 5) ** 2:  # 5 pixels margin
                                # Initiate dragging the node
                                dragged_node = node
                                node.fixed = True
                                break  # Only one node can be interacted with at a time
                    else:
                        # If drag mode is inactive, handle source node selection
                        for idx, node in enumerate(nodes):
                            # Transform node position to screen coordinates
                            node_screen_x = (node.x + offset_x) * zoom
                            node_screen_y = (node.y + offset_y) * zoom
                            # Define node radius for clicking
                            node_radius = 6
                            distance_sq = (mouse_x - node_screen_x) ** 2 + (mouse_y - node_screen_y) ** 2
                            if distance_sq <= (node_radius + 5) ** 2:  # 5 pixels margin
                                # Set the source node
                                source_node = node
                                # Reset wave_step
                                wave_step = 0
                                wave_vector = None
                                # Reset node colors
                                for n in nodes:
                                    n.color = GRAY
                                # Highlight the source node
                                source_node.color = PURPLE
                                break  # Only one node can be interacted with at a time

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button released
                    dragging_speed = False
                    dragging_zoom = False
                    dragging_wave = False
                    if dragged_node:
                        dragged_node.fixed = False
                        dragged_node = None

            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = event.pos
                if dragging_speed:
                    speed_slider_x = 50
                    speed_slider_width = 200
                    x = min(max(mouse_x, speed_slider_x), speed_slider_x + speed_slider_width)
                    slider_position = (x - speed_slider_x) / speed_slider_width
                    iterations_per_frame = max(MIN_ITERATIONS, min(int(MIN_ITERATIONS + slider_position * (MAX_ITERATIONS - MIN_ITERATIONS)), MAX_ITERATIONS))
                if dragging_zoom:
                    zoom_slider_x = 300
                    zoom_slider_width = 200
                    x = min(max(mouse_x, zoom_slider_x), zoom_slider_x + zoom_slider_width)
                    slider_position = (x - zoom_slider_x) / zoom_slider_width
                    new_zoom_level = MIN_ZOOM + slider_position * (MAX_ZOOM - MIN_ZOOM)
                    
                    # Center the zoom on the screen center
                    cx, cy = WIDTH / 2, (HEIGHT - PANEL_HEIGHT) / 2
                    # Calculate world coordinates before zoom
                    wx = (cx / zoom) - offset_x
                    wy = (cy / zoom) - offset_y
                    # Update zoom level
                    zoom_level = new_zoom_level
                    zoom = zoom_level
                    # Calculate new offsets to keep (wx, wy) at the center
                    offset_x = (cx / zoom) - wx
                    offset_y = (cy / zoom) - wy
                if dragging_wave:
                    wave_slider_x = 550
                    wave_slider_width = 200
                    x = min(max(mouse_x, wave_slider_x), wave_slider_x + wave_slider_width)
                    slider_position = (x - wave_slider_x) / wave_slider_width
                    wave_step = max(MIN_WAVE_STEP, min(int(MIN_WAVE_STEP + slider_position * (MAX_WAVE_STEP - MIN_WAVE_STEP)), MAX_WAVE_STEP))

                # Update position of the dragged node
                if dragged_node:
                    # Convert mouse position to world coordinates
                    world_x = (mouse_x / zoom) - offset_x
                    world_y = (mouse_y / zoom) - offset_y
                    dragged_node.x = world_x
                    dragged_node.y = world_y

            elif event.type == pygame.KEYDOWN:
                active_field = None
                if input_active['num_nodes']:
                    active_field = 'num_nodes'
                elif input_active['edge_prob']:
                    active_field = 'edge_prob'
                
                if active_field:
                    if event.key == pygame.K_BACKSPACE:
                        input_boxes[active_field]['text'] = input_boxes[active_field]['text'][:-1]
                    elif event.key == pygame.K_RETURN:
                        input_active['num_nodes'] = False
                        input_active['edge_prob'] = False
                    else:
                        # Limit input length
                        if len(input_boxes[active_field]['text']) < 10:
                            input_boxes[active_field]['text'] += event.unicode

        # Handle hover state for refresh and drag mode buttons
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if input_boxes['refresh_button']['rect'].collidepoint((mouse_x, mouse_y)):
            refresh_button_hover = True
        else:
            refresh_button_hover = False

        if input_boxes['drag_mode_button']['rect'].collidepoint((mouse_x, mouse_y)):
            drag_mode_button_hover = True
        else:
            drag_mode_button_hover = False

        # Handle panning with left mouse button (when not dragging a node)
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0]:  # Left mouse button
            # Check if not clicking on sliders or input fields or buttons
            if not ((50 <= mouse_x <= 250 and HEIGHT - PANEL_HEIGHT + 20 <= mouse_y <= HEIGHT - PANEL_HEIGHT + 80) or
                    (300 <= mouse_x <= 500 and HEIGHT - PANEL_HEIGHT + 20 <= mouse_y <= HEIGHT - PANEL_HEIGHT + 80) or
                    (550 <= mouse_x <= 750 and HEIGHT - PANEL_HEIGHT + 20 <= mouse_y <= HEIGHT - PANEL_HEIGHT + 80) or
                    input_boxes['num_nodes']['rect'].collidepoint((mouse_x, mouse_y)) or
                    input_boxes['edge_prob']['rect'].collidepoint((mouse_x, mouse_y)) or
                    input_boxes['refresh_button']['rect'].collidepoint((mouse_x, mouse_y)) or
                    input_boxes['drag_mode_button']['rect'].collidepoint((mouse_x, mouse_y))):
                if not dragging_speed and not dragging_zoom and not dragging_wave and not dragged_node:
                    if not panning:
                        panning = True
                        pan_start_pos = (mouse_x, mouse_y)
                    else:
                        dx = mouse_x - pan_start_pos[0]
                        dy = mouse_y - pan_start_pos[1]
                        offset_x += dx / zoom
                        offset_y += dy / zoom
                        pan_start_pos = (mouse_x, mouse_y)
        else:
            panning = False

        # Perform multiple iterations per frame
        for _ in range(iterations_per_frame):
            apply_forces(nodes, adj_matrix)

        # Handle wave propagation
        if wave_step > 0 and source_node is not None:
            source_index = nodes.index(source_node)
            # Initialize the initial wave vector
            if wave_step == 1:
                wave_vector = markov_matrix[source_index]
            else:
                wave_vector = np.linalg.matrix_power(markov_matrix, wave_step)[source_index]
            # Update node colors based on wave_vector
            for idx, node in enumerate(nodes):
                if wave_vector[idx] > 0:
                    # Set color based on probability (0 to 1)
                    node.color = probability_to_color(wave_vector[idx])
                else:
                    if node != source_node:
                        node.color = GRAY
            # Ensure the source node is highlighted
            source_node.color = PURPLE

        # Compute energy after force application
        energy = compute_energy(nodes, adj_matrix)

        # Clear the screen and draw the graph and UI elements
        draw_graph(screen, nodes, adj_matrix, zoom, offset_x, offset_y, dragged_node, source_node, wave_step, wave_vector)
        draw_ui(screen, iterations_per_frame, zoom_level, 
                int(input_boxes['num_nodes']['text']) if input_boxes['num_nodes']['text'].isdigit() else 0, 
                float(input_boxes['edge_prob']['text']) if input_boxes['edge_prob']['text'].replace('.', '', 1).isdigit() else 0.0,
                energy,
                input_active, input_boxes, refresh_button_hover, drag_mode, drag_mode_button_hover, wave_step, False)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
