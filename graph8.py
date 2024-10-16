import pygame
import math
import random
import heapq  # Added for priority queue in Dijkstra's algorithm

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 750  # Increased height for a larger panel
PANEL_HEIGHT = 200         # Increased panel height to accommodate new UI elements
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Animated Force-Directed Graph with Dijkstra Wavefront")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
DARK_GRAY = (50, 50, 50)
LIGHT_BLUE = (173, 216, 230)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)   # Color for dragged node
ORANGE = (255, 165, 0)   # Color for nodes in queue
PURPLE = (128, 0, 128)   # Distinct color for the source node
BUTTON_COLOR = (70, 130, 180)  # Steel Blue for buttons
BUTTON_HOVER_COLOR = (100, 149, 237)  # Cornflower Blue for button hover

# Slider parameters for speed control
MIN_ITERATIONS = 1
MAX_ITERATIONS = 100  # Allows up to 100 iterations per frame
iterations_per_frame = 1  # Default iterations per frame

# Slider parameters for zoom control
MIN_ZOOM = 0.1  # Increased zoom-out capability
MAX_ZOOM = 2.0  # Limited zoom-in to prevent clutter
zoom_level = 1.0  # Default zoom level

# Slider parameters for Max Distance control
MIN_DISTANCE = 0
MAX_DISTANCE = 20  # You can adjust this based on your graph size
current_max_distance = MAX_DISTANCE  # Default value

# Physics parameters
REPULSION_STRENGTH = 5000  # Increased strength for better separation
ATTRACTION_STRENGTH = 0.01  # Strength of the attraction along edges
DAMPING = 0.9  # Damping factor to reduce velocity over time
TIME_STEP = 0.1  # Small time step for accurate simulation

# Font setup for slider text and UI elements
FONT = pygame.font.SysFont(None, 24)
FONT_SMALL = pygame.font.SysFont(None, 20)

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
        self.fixed = False  # Indicates if the node is being dragged

        # Attributes for Dijkstra's traversal
        self.visited = False
        self.distance = float('inf')  # Tentative distance from source
        self.previous = None  # Previous node in the optimal path
        self.in_queue = False  # Indicates if the node is in the priority queue

# Generate random graph
def generate_random_graph(num_nodes, edge_probability):
    nodes = []
    margin = 50  # Margin to prevent nodes from being too close to the edges
    for _ in range(num_nodes):
        x = random.randint(margin, WIDTH - margin)
        y = random.randint(margin, HEIGHT - PANEL_HEIGHT - margin)
        nodes.append(Node(x, y))
    
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
        node.fx = 0.0
        node.fy = 0.0

    # Repulsion between nodes
    for i, node1 in enumerate(nodes):
        if node1.fixed:
            continue  # Skip force calculations for fixed nodes
        for j in range(i + 1, len(nodes)):
            node2 = nodes[j]
            if node2.fixed:
                continue  # Skip force calculations for fixed nodes
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
        if node.fixed:
            continue  # Skip force calculations for fixed nodes
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
        if node.fixed:
            node.vx = 0.0  # Reset velocity for fixed nodes
            node.vy = 0.0
            continue  # Skip position update for fixed nodes
        node.vx = (node.vx + node.fx * TIME_STEP) * DAMPING
        node.vy = (node.vy + node.fy * TIME_STEP) * DAMPING
        node.x += node.vx * TIME_STEP
        node.y += node.vy * TIME_STEP

        # No clamping to allow nodes to move off-screen

# Compute the total potential energy of the system
def compute_energy(nodes):
    total_energy = 0.0

    # Repulsive energy between all node pairs
    for i, node1 in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            node2 = nodes[j]
            dx = node1.x - node2.x
            dy = node1.y - node2.y
            distance = math.hypot(dx, dy)
            if distance == 0:
                continue  # Avoid division by zero
            repulsion_energy = REPULSION_STRENGTH / distance
            total_energy += repulsion_energy

    # Attractive energy along edges
    for node_idx, node in enumerate(nodes):
        for edge_index in node.edges:
            if edge_index < node_idx:
                continue  # Avoid double processing
            other_node = nodes[edge_index]
            dx = other_node.x - node.x
            dy = other_node.y - node.y
            distance = math.hypot(dx, dy)
            if distance == 0:
                continue  # Avoid division by zero
            attraction_energy = ATTRACTION_STRENGTH * distance * distance / 2
            total_energy += attraction_energy

    return total_energy

# Draw the graph with zoom and pan
def draw_graph(screen, nodes, zoom, offset_x, offset_y, dragged_node, current_max_distance, source_node):
    # Define the drawing area (excluding the panel)
    drawing_area_rect = pygame.Rect(0, 0, WIDTH, HEIGHT - PANEL_HEIGHT)
    pygame.draw.rect(screen, WHITE, drawing_area_rect)

    # Draw edges
    for node_idx, node in enumerate(nodes):
        for edge_index in node.edges:
            # To prevent drawing the same edge twice
            if edge_index < node_idx:
                continue
            other_node = nodes[edge_index]
            # Apply zoom and offset
            x1 = int((node.x + offset_x) * zoom)
            y1 = int((node.y + offset_y) * zoom)
            x2 = int((other_node.x + offset_x) * zoom)
            y2 = int((other_node.y + offset_y) * zoom)
            pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), 1)

    # Draw nodes
    for node in nodes:
        # Apply zoom and offset
        x = (node.x + offset_x) * zoom
        y = (node.y + offset_y) * zoom
        # Only draw nodes that are within the visible screen area
        if 0 <= x <= WIDTH and 0 <= y <= (HEIGHT - PANEL_HEIGHT):
            # Determine node color based on traversal state and max distance
            if node.distance <= current_max_distance:
                if node == source_node:
                    color = PURPLE  # Distinct color for the source node
                elif node.visited:
                    color = GREEN
                elif node.in_queue:
                    color = ORANGE
                else:
                    color = RED
            else:
                color = RED  # Default color for nodes beyond max distance

            # Override color if the node is being dragged
            if node == dragged_node:
                color = YELLOW

            pygame.draw.circle(screen, color, (int(x), int(y)), 6 if node == dragged_node else 4)
            
            # Draw a circle around the source node
            if node == source_node:
                pygame.draw.circle(screen, WHITE, (int(x), int(y)), 10, 2)  # White circle with radius 10 and thickness 2

# Draw the sliders and UI elements within the panel
def draw_ui(screen, iterations_per_frame, zoom_level, num_nodes, edge_prob, energy, input_active, input_boxes, refresh_button_hover, drag_mode, drag_mode_hover, current_max_distance):
    # Define the panel area
    panel_rect = pygame.Rect(0, HEIGHT - PANEL_HEIGHT, WIDTH, PANEL_HEIGHT)
    pygame.draw.rect(screen, DARK_GRAY, panel_rect)

    # Slider 1: Speed Control
    speed_slider_x = 50
    speed_slider_y = HEIGHT - PANEL_HEIGHT + 50
    speed_slider_width = 300
    speed_slider_height = 20

    # Slider background
    pygame.draw.rect(screen, GRAY, (speed_slider_x, speed_slider_y, speed_slider_width, speed_slider_height))

    # Slider handle
    speed_slider_position = (iterations_per_frame - MIN_ITERATIONS) / (MAX_ITERATIONS - MIN_ITERATIONS)
    speed_handle_x = speed_slider_x + int(speed_slider_position * speed_slider_width)
    speed_handle_y = speed_slider_y + speed_slider_height // 2
    pygame.draw.circle(screen, BLUE, (speed_handle_x, speed_handle_y), 10)

    # Slider 2: Zoom Control
    zoom_slider_x = 450
    zoom_slider_y = HEIGHT - PANEL_HEIGHT + 50
    zoom_slider_width = 300
    zoom_slider_height = 20

    # Slider background
    pygame.draw.rect(screen, GRAY, (zoom_slider_x, zoom_slider_y, zoom_slider_width, zoom_slider_height))

    # Slider handle
    zoom_slider_position = (zoom_level - MIN_ZOOM) / (MAX_ZOOM - MIN_ZOOM)
    zoom_handle_x = zoom_slider_x + int(zoom_slider_position * zoom_slider_width)
    zoom_handle_y = zoom_slider_y + zoom_slider_height // 2
    pygame.draw.circle(screen, BLUE, (zoom_handle_x, zoom_handle_y), 10)

    # Slider 3: Max Distance Control
    distance_slider_x = 50
    distance_slider_y = HEIGHT - PANEL_HEIGHT + 150
    distance_slider_width = 300
    distance_slider_height = 20

    # Slider background
    pygame.draw.rect(screen, GRAY, (distance_slider_x, distance_slider_y, distance_slider_width, distance_slider_height))

    # Slider handle
    distance_slider_position = (current_max_distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    distance_handle_x = distance_slider_x + int(distance_slider_position * distance_slider_width)
    distance_handle_y = distance_slider_y + distance_slider_height // 2
    pygame.draw.circle(screen, BLUE, (distance_handle_x, distance_handle_y), 10)

    # Draw slider labels
    label_speed = FONT.render("Speed Control:", True, WHITE)
    screen.blit(label_speed, (speed_slider_x, speed_slider_y - 30))

    label_zoom = FONT.render("Zoom Control:", True, WHITE)
    screen.blit(label_zoom, (zoom_slider_x, zoom_slider_y - 30))

    label_distance = FONT.render("Max Distance:", True, WHITE)
    screen.blit(label_distance, (distance_slider_x, distance_slider_y - 30))

    # Draw current slider values
    speed_text = FONT.render(f"Speed: {iterations_per_frame}x", True, WHITE)
    screen.blit(speed_text, (speed_slider_x + speed_slider_width // 2 - speed_text.get_width() // 2, speed_slider_y - 30))

    zoom_display = f"{zoom_level:.2f}x"
    zoom_text = FONT.render(f"Zoom: {zoom_display}", True, WHITE)
    screen.blit(zoom_text, (zoom_slider_x + zoom_slider_width // 2 - zoom_text.get_width() // 2, zoom_slider_y - 30))

    distance_text = FONT.render(f"Distance: {current_max_distance}", True, WHITE)
    screen.blit(distance_text, (distance_slider_x + distance_slider_width // 2 - distance_text.get_width() // 2, distance_slider_y - 30))

    # Input Fields: Number of Nodes and Edge Probability
    # Number of Nodes
    nodes_label = FONT.render("Number of Nodes:", True, WHITE)
    screen.blit(nodes_label, (50, HEIGHT - PANEL_HEIGHT + 100))
    pygame.draw.rect(screen, LIGHT_BLUE if input_active['num_nodes'] else GRAY, input_boxes['num_nodes']['rect'], 2)
    nodes_text = FONT.render(input_boxes['num_nodes']['text'], True, BLACK)
    screen.blit(nodes_text, (input_boxes['num_nodes']['rect'].x + 5, input_boxes['num_nodes']['rect'].y + 5))

    # Edge Probability
    edge_label = FONT.render("Edge Probability (0-1):", True, WHITE)
    screen.blit(edge_label, (450, HEIGHT - PANEL_HEIGHT + 100))
    pygame.draw.rect(screen, LIGHT_BLUE if input_active['edge_prob'] else GRAY, input_boxes['edge_prob']['rect'], 2)
    edge_text = FONT.render(input_boxes['edge_prob']['text'], True, BLACK)
    screen.blit(edge_text, (input_boxes['edge_prob']['rect'].x + 5, input_boxes['edge_prob']['rect'].y + 5))

    # Refresh Button
    refresh_button_rect = input_boxes['refresh_button']['rect']
    pygame.draw.rect(screen, GREEN if refresh_button_hover else GRAY, refresh_button_rect)
    refresh_text = FONT.render("Refresh", True, BLACK)
    screen.blit(refresh_text, (refresh_button_rect.x + (refresh_button_rect.width - refresh_text.get_width()) // 2,
                               refresh_button_rect.y + (refresh_button_rect.height - refresh_text.get_height()) // 2))

    # Drag Mode Toggle Button
    drag_mode_button_rect = input_boxes['drag_mode_button']['rect']
    button_color = BUTTON_HOVER_COLOR if drag_mode_hover else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, drag_mode_button_rect)
    drag_mode_text = FONT.render(f"Drag Mode: {'ON' if drag_mode else 'OFF'}", True, WHITE)
    screen.blit(drag_mode_text, (drag_mode_button_rect.x + (drag_mode_button_rect.width - drag_mode_text.get_width()) // 2,
                                 drag_mode_button_rect.y + (drag_mode_button_rect.height - drag_mode_text.get_height()) // 2))

    # Display Energy
    energy_text = FONT.render(f"Energy: {energy:.2f}", True, WHITE)
    screen.blit(energy_text, (350, HEIGHT - PANEL_HEIGHT + 150))

    # Instructions
    instructions = [
        "Click on a node to start Dijkstra's traversal.",
        "Visited nodes are Green, In-Queue nodes are Orange.",
        "Toggle Drag Mode to drag nodes."
    ]
    for idx, text in enumerate(instructions):
        instr_text = FONT_SMALL.render(text, True, WHITE)
        screen.blit(instr_text, (50, HEIGHT - PANEL_HEIGHT + 180 + idx * 20))

# Main loop
def main():
    global iterations_per_frame, zoom_level
    clock = pygame.time.Clock()
    running = True
    dragging_speed = False     # To track if the speed slider is being dragged
    dragging_zoom = False      # To track if the zoom slider is being dragged
    dragging_distance = False  # To track if the distance slider is being dragged
    panning = False            # To track if panning is active
    pan_start_pos = (0, 0)     # Starting position for panning
    dragged_node = None        # Currently dragged node

    # Zoom and Pan parameters
    zoom = zoom_level
    offset_x = 0
    offset_y = 0

    # Dijkstra's traversal state
    traversal_active = False
    traversal_queue = []
    source_node = None

    # Generate the initial graph as a random graph
    nodes = generate_random_graph(50, 0.05)  # Default: 50 nodes, 5% edge probability

    # UI State
    input_active = {'num_nodes': False, 'edge_prob': False}
    input_boxes = {
        'num_nodes': {'rect': pygame.Rect(200, HEIGHT - PANEL_HEIGHT + 100, 200, 30), 'text': '50'},
        'edge_prob': {'rect': pygame.Rect(650, HEIGHT - PANEL_HEIGHT + 100, 100, 30), 'text': '0.05'},
        'refresh_button': {'rect': pygame.Rect(350, HEIGHT - PANEL_HEIGHT + 100, 80, 30)},
        'drag_mode_button': {'rect': pygame.Rect(350, HEIGHT - PANEL_HEIGHT + 150, 150, 30)}  # Added Drag Mode Button
    }
    refresh_button_hover = False
    drag_mode_button_hover = False  # Hover state for Drag Mode Button

    # Max Distance Slider
    distance_slider_x = 50
    distance_slider_y = HEIGHT - PANEL_HEIGHT + 150
    distance_slider_width = 300
    distance_slider_height = 20
    current_max_distance = MAX_DISTANCE  # Initialize with MAX_DISTANCE

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
                speed_slider_width = 300
                speed_slider_height = 20
                handle_radius = 10
                if (speed_slider_x <= mouse_x <= speed_slider_x + speed_slider_width) and \
                   (speed_slider_y - handle_radius <= mouse_y <= speed_slider_y + speed_slider_height + handle_radius):
                    dragging_speed = True
                    # Update the iterations_per_frame based on mouse position
                    x = min(max(mouse_x, speed_slider_x), speed_slider_x + speed_slider_width)
                    slider_position = (x - speed_slider_x) / speed_slider_width
                    iterations_per_frame = int(MIN_ITERATIONS + slider_position * (MAX_ITERATIONS - MIN_ITERATIONS))

                # Check if the click is within the zoom slider area
                zoom_slider_x = 450
                zoom_slider_y = HEIGHT - PANEL_HEIGHT + 50
                zoom_slider_width = 300
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

                # Check if the click is within the distance slider area
                if (distance_slider_x <= mouse_x <= distance_slider_x + distance_slider_width) and \
                   (distance_slider_y - handle_radius <= mouse_y <= distance_slider_y + distance_slider_height + handle_radius):
                    dragging_distance = True
                    # Update the current_max_distance based on mouse position
                    x = min(max(mouse_x, distance_slider_x), distance_slider_x + distance_slider_width)
                    slider_position = (x - distance_slider_x) / distance_slider_width
                    current_max_distance = int(MIN_DISTANCE + slider_position * (MAX_DISTANCE - MIN_DISTANCE))
                    # Clamp the value
                    current_max_distance = max(MIN_DISTANCE, min(current_max_distance, MAX_DISTANCE))

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
                        nodes = generate_random_graph(num_nodes, edge_prob)
                        # Reset traversal state
                        traversal_active = False
                        traversal_queue = []
                        source_node = None
                        current_max_distance = MAX_DISTANCE  # Reset max distance
                        for node in nodes:
                            node.visited = False
                            node.distance = float('inf')
                            node.previous = None
                            node.in_queue = False
                    except ValueError:
                        print("Invalid input for number of nodes or edge probability.")

                # Check if click is on the Drag Mode button
                if input_boxes['drag_mode_button']['rect'].collidepoint(event.pos):
                    drag_mode = not drag_mode  # Toggle drag mode
                else:
                    # Check if click is on a node based on the current mode
                    for idx, node in enumerate(nodes):
                        # Transform node position to screen coordinates
                        node_screen_x = (node.x + offset_x) * zoom
                        node_screen_y = (node.y + offset_y) * zoom
                        # Define node radius for clicking
                        node_radius = 6 if node == dragged_node else 4
                        distance_sq = (mouse_x - node_screen_x) ** 2 + (mouse_y - node_screen_y) ** 2
                        if distance_sq <= (node_radius + 5) ** 2:  # 5 pixels margin
                            if drag_mode:
                                # Initiate dragging the node
                                dragged_node = node
                                node.fixed = True
                            else:
                                # Initiate traversal if not dragging
                                if not traversal_active:
                                    # Reset traversal state
                                    for n in nodes:
                                        n.visited = False
                                        n.distance = float('inf')
                                        n.previous = None
                                        n.in_queue = False
                                    # Initialize source node
                                    source_node = node
                                    source_node.distance = 0
                                    heapq.heappush(traversal_queue, (source_node.distance, idx))
                                    source_node.in_queue = True
                                    traversal_active = True
                            break  # Only one node can be interacted with at a time

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_speed = False
                    dragging_zoom = False
                    dragging_distance = False
                    if dragged_node:
                        dragged_node.fixed = False
                        dragged_node = None

            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = event.pos
                if dragging_speed:
                    speed_slider_x = 50
                    speed_slider_width = 300
                    x = min(max(mouse_x, speed_slider_x), speed_slider_x + speed_slider_width)
                    slider_position = (x - speed_slider_x) / speed_slider_width
                    iterations_per_frame = int(MIN_ITERATIONS + slider_position * (MAX_ITERATIONS - MIN_ITERATIONS))
                if dragging_zoom:
                    zoom_slider_x = 450
                    zoom_slider_width = 300
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
                if dragging_distance:
                    x = min(max(mouse_x, distance_slider_x), distance_slider_x + distance_slider_width)
                    slider_position = (x - distance_slider_x) / distance_slider_width
                    current_max_distance = int(MIN_DISTANCE + slider_position * (MAX_DISTANCE - MIN_DISTANCE))
                    # Clamp the value
                    current_max_distance = max(MIN_DISTANCE, min(current_max_distance, MAX_DISTANCE))

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

        # Handle panning with left mouse button
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0]:  # Left mouse button
            # Check if not clicking on sliders or input fields or buttons
            if not ((50 <= mouse_x <= 350 and HEIGHT - PANEL_HEIGHT + 20 <= mouse_y <= HEIGHT - PANEL_HEIGHT + 80) or
                    (450 <= mouse_x <= 750 and HEIGHT - PANEL_HEIGHT + 20 <= mouse_y <= HEIGHT - PANEL_HEIGHT + 80) or
                    (distance_slider_x <= mouse_x <= distance_slider_x + distance_slider_width and
                     distance_slider_y - 10 <= mouse_y <= distance_slider_y + distance_slider_height + 10) or
                    input_boxes['num_nodes']['rect'].collidepoint((mouse_x, mouse_y)) or
                    input_boxes['edge_prob']['rect'].collidepoint((mouse_x, mouse_y)) or
                    input_boxes['refresh_button']['rect'].collidepoint((mouse_x, mouse_y)) or
                    input_boxes['drag_mode_button']['rect'].collidepoint((mouse_x, mouse_y))):
                if not dragging_speed and not dragging_zoom and not dragging_distance and not dragged_node:
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
            apply_forces(nodes)

        # Perform Dijkstra's traversal steps
        if traversal_active and traversal_queue:
            current_distance, current_idx = heapq.heappop(traversal_queue)
            current_node = nodes[current_idx]
            current_node.in_queue = False

            if not current_node.visited:
                current_node.visited = True
                # Process all adjacent nodes
                for neighbor_idx in current_node.edges:
                    neighbor = nodes[neighbor_idx]
                    if not neighbor.visited:
                        new_distance = current_node.distance + 1  # Assuming unit weight
                        if new_distance < neighbor.distance:
                            neighbor.distance = new_distance
                            neighbor.previous = current_idx
                            if not neighbor.in_queue:
                                heapq.heappush(traversal_queue, (neighbor.distance, neighbor_idx))
                                neighbor.in_queue = True

            # Check if traversal is complete
            if not traversal_queue:
                traversal_active = False

        # Compute energy after force application
        energy = compute_energy(nodes)

        # Clear the screen and draw the graph and UI elements
        draw_graph(screen, nodes, zoom, offset_x, offset_y, dragged_node, current_max_distance, source_node)
        draw_ui(screen, iterations_per_frame, zoom_level, 
                int(input_boxes['num_nodes']['text']) if input_boxes['num_nodes']['text'].isdigit() else 0, 
                float(input_boxes['edge_prob']['text']) if input_boxes['edge_prob']['text'].replace('.', '', 1).isdigit() else 0.0,
                energy,
                input_active, input_boxes, refresh_button_hover, drag_mode, drag_mode_button_hover, current_max_distance)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
