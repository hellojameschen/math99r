import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
PANEL_HEIGHT = 150  # Adjusted to accommodate sliders and buttons
screen = pygame.display.set_mode((WIDTH, HEIGHT + PANEL_HEIGHT))
pygame.display.set_caption('Lorenz Attractor Simulation')

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SLIDER_COLOR = (200, 200, 200)
SLIDER_HANDLE_COLOR = (100, 100, 100)

# Font
font = pygame.font.SysFont(None, 18)

# Lorenz attractor default parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
dt = 0.01

# Initialize variables
x = 0.1
y = 0.0
z = 0.0

# Scale and offset for visualization
scale = 10
offset_x = WIDTH // 2
offset_y = HEIGHT // 2

# List to store points (will store (x, y, z) coordinates)
points = []

def draw_slider(screen, x, y, width, height, value, color, label, min_value, max_value):
    # Draw the slider background
    pygame.draw.rect(screen, SLIDER_COLOR, (x, y, width, height))
    # Draw the slider handle
    handle_x = x + (value - min_value) / (max_value - min_value) * width
    pygame.draw.circle(screen, color, (int(handle_x), y + height // 2), 8)
    # Render the label
    label_surface = font.render(f"{label}: {value:.2f}", True, BLACK)
    screen.blit(label_surface, (x, y - 20))

def handle_slider_event(mouse_pos, x, y, width, height, min_value, max_value):
    if x <= mouse_pos[0] <= x + width and y <= mouse_pos[1] <= y + height:
        relative_position = (mouse_pos[0] - x) / width
        value = min_value + relative_position * (max_value - min_value)
        return value
    return None

def draw_button(screen, rect, text):
    pygame.draw.rect(screen, SLIDER_COLOR, rect)
    pygame.draw.rect(screen, BLACK, rect, 2)
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

def update_offsets():
    global offset_x, offset_y
    # Calculate the mean positions of the attractor's lobes
    mean_x = 0  # The Lorenz attractor is symmetric around x=0
    mean_z = rho - 1
    offset_x = WIDTH // 2 - mean_x * scale
    offset_y = HEIGHT // 2 - (-mean_z * scale)  # Invert z for Pygame coordinates

def reset_simulation():
    global x, y, z, points
    x, y, z = 0.1, 0.0, 0.0
    points = []
    update_offsets()

# Run the simulation
clock = pygame.time.Clock()
running = True
paused = False

# Positions for sliders
slider_positions = {
    'sigma': {'x': 50, 'y': HEIGHT + 30, 'min': 0.0, 'max': 50.0, 'value': sigma},
    'rho': {'x': 300, 'y': HEIGHT + 30, 'min': 0.0, 'max': 50.0, 'value': rho},
    'beta': {'x': 550, 'y': HEIGHT + 30, 'min': 0.0, 'max': 10.0, 'value': beta},
    'scale': {'x': 50, 'y': HEIGHT + 80, 'min': 1.0, 'max': 50.0, 'value': scale},
    'dt': {'x': 300, 'y': HEIGHT + 80, 'min': 0.001, 'max': 0.02, 'value': dt},
}

# Buttons
pause_button_rect = pygame.Rect(550, HEIGHT + 70, 80, 30)
reset_button_rect = pygame.Rect(650, HEIGHT + 70, 80, 30)

# Initial centering of the attractor
update_offsets()

while running:
    clock.tick(60)  # Limit to 60 FPS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            # Handle sliders
            if mouse_pos[1] > HEIGHT:
                for key, pos in slider_positions.items():
                    new_value = handle_slider_event(
                        mouse_pos, pos['x'], pos['y'], 200, 10, pos['min'], pos['max']
                    )
                    if new_value is not None:
                        pos['value'] = new_value
                        # Update parameter values
                        if key == 'sigma':
                            sigma = new_value
                        elif key == 'rho':
                            rho = new_value
                        elif key == 'beta':
                            beta = new_value
                        elif key == 'scale':
                            scale = new_value
                        elif key == 'dt':
                            dt = new_value
                        # After updating parameters, update offsets
                        update_offsets()
                # Pause button
                if pause_button_rect.collidepoint(mouse_pos):
                    paused = not paused
                # Reset button
                if reset_button_rect.collidepoint(mouse_pos):
                    reset_simulation()

    if not paused:
        # Update the Lorenz attractor equations
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt

        x += dx
        y += dy
        z += dz

        # Store the raw (x, y, z) coordinates
        points.append((x, y, z))

    # Draw everything
    screen.fill(BLACK)

    # Draw the points
    for point in points:
        x_p, y_p, z_p = point
        screen_x = int(x_p * scale + offset_x)
        screen_y = int(-z_p * scale + offset_y)  # Invert z to correct the direction
        if 0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT:
            screen.set_at((screen_x, screen_y), WHITE)

    # Draw the panel background
    pygame.draw.rect(screen, WHITE, (0, HEIGHT, WIDTH, PANEL_HEIGHT))

    # Draw sliders
    for key, pos in slider_positions.items():
        draw_slider(
            screen, pos['x'], pos['y'], 200, 10, pos['value'], SLIDER_HANDLE_COLOR, key.capitalize(), pos['min'], pos['max']
        )

    # Draw buttons
    draw_button(screen, pause_button_rect, 'Resume' if paused else 'Pause')
    draw_button(screen, reset_button_rect, 'Reset')

    pygame.display.flip()

pygame.quit()
