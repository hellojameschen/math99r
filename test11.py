import pygame
import numpy as np

# Initialize Pygame
pygame.init()

BOUNCE = True
G = 6.67430e-11  # Gravitational constant
scale = 1e8      # Scale factor for distances
mouse_velocity_scale = 0.001  # Scale factor for the mouse velocity

# Set up display
WIDTH, HEIGHT = 800, 800
PANEL_HEIGHT = 500  # Increased to accommodate additional info
screen = pygame.display.set_mode((WIDTH, HEIGHT + PANEL_HEIGHT))
pygame.display.set_caption('Three-Body Problem with Trails')

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BODY_COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
SLIDER_COLOR = (200, 200, 200)
SLIDER_HANDLE_COLOR = (100, 100, 100)

class Body:
    def __init__(self, mass, position, velocity, color):
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.color = color
        self.is_dragged = False
        self.prev_mouse_pos = None  # For momentum calculation
        self.mouse_velocity = np.zeros(2)
        self.positions = []  # List to store trail points with alpha

def gravitational_force(body1, body2):
    # Calculate the gravitational force between two bodies
    r_vec = body2.position - body1.position
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros(2)
    force_magnitude = G * body1.mass * body2.mass / r**2
    force_direction = r_vec / r
    return force_magnitude * force_direction

def update_velocity(bodies, dt):
    for i, body in enumerate(bodies):
        if not body.is_dragged:
            force = np.zeros(2)
            for j, other_body in enumerate(bodies):
                if i != j:
                    force += gravitational_force(body, other_body)
            acceleration = force / body.mass
            body.velocity += acceleration * dt

def update_position(bodies, dt, offset_x, offset_y):
    for body in bodies:
        if not body.is_dragged:
            body.position += body.velocity * dt

            if BOUNCE:
                x_screen, y_screen = transform_position(body.position, offset_x, offset_y)
                bounced = False
                RADIUS = 10  # Radius of the bodies

                if x_screen <= RADIUS:
                    body.velocity[0] = -body.velocity[0]
                    x_screen = RADIUS
                    bounced = True
                elif x_screen >= WIDTH - RADIUS:
                    body.velocity[0] = -body.velocity[0]
                    x_screen = WIDTH - RADIUS
                    bounced = True

                if y_screen <= RADIUS:
                    body.velocity[1] = -body.velocity[1]
                    y_screen = RADIUS
                    bounced = True
                elif y_screen >= HEIGHT - RADIUS:
                    body.velocity[1] = -body.velocity[1]
                    y_screen = HEIGHT - RADIUS
                    bounced = True

                if bounced:
                    body.position = inverse_transform_position(x_screen, y_screen, offset_x, offset_y)

def draw_bodies(bodies, screen, trail_surface, offset_x, offset_y):
    screen.fill(BLACK, (0, 0, WIDTH, HEIGHT))  # Clear the simulation area
    trail_surface.fill((0, 0, 0, 0))  # Clear the trail surface

    for body in bodies:
        # Draw the trail
        for trail_point in body.positions:
            x, y = transform_position(trail_point['position'], offset_x, offset_y)
            alpha = max(0, min(255, trail_point['alpha']))
            color_with_alpha = body.color + (int(alpha),)
            pygame.draw.circle(trail_surface, color_with_alpha, (int(x), int(y)), 2)

        # Draw the body
        x, y = transform_position(body.position, offset_x, offset_y)
        pygame.draw.circle(screen, body.color, (int(x), int(y)), 10)

    # Blit the trail surface onto the main screen
    screen.blit(trail_surface, (0, 0))

def transform_position(position, offset_x, offset_y):
    x = WIDTH // 2 + (position[0] - offset_x) / scale
    y = HEIGHT // 2 + (position[1] - offset_y) / scale
    return x, y

def inverse_transform_position(x, y, offset_x, offset_y):
    pos_x = (x - WIDTH // 2) * scale + offset_x
    pos_y = (y - HEIGHT // 2) * scale + offset_y
    return np.array([pos_x, pos_y])

def is_mouse_over_body(body, mouse_pos, offset_x, offset_y):
    x, y = transform_position(body.position, offset_x, offset_y)
    distance = np.linalg.norm(np.array(mouse_pos) - np.array([x, y]))
    return distance < 10  # 10 is the radius of the body

def draw_panel(screen, bodies, trail_fade_speed):
    # Draw the panel background
    pygame.draw.rect(screen, WHITE, (0, HEIGHT, WIDTH, PANEL_HEIGHT))

    # Set up the font
    font = pygame.font.SysFont(None, 20)

    # Draw sliders and display information for each body
    for i, body in enumerate(bodies):
        slider_x = 100 + i * 250
        slider_y = HEIGHT + 20
        draw_slider(screen, slider_x, slider_y, 200, 10, body.mass / 1e30, BODY_COLOR[i], f"Mass {i+1}")

        # Display mass
        mass_text = font.render(f"Mass: {body.mass:.2e} kg", True, BLACK)
        screen.blit(mass_text, (slider_x, slider_y + 15))

        # Display position
        pos_text = font.render(f"Pos: ({body.position[0]:.2e}, {body.position[1]:.2e}) m", True, BLACK)
        screen.blit(pos_text, (slider_x, slider_y + 35))

        # Display velocity
        vel_text = font.render(f"Vel: ({body.velocity[0]:.2e}, {body.velocity[1]:.2e}) m/s", True, BLACK)
        screen.blit(vel_text, (slider_x, slider_y + 55))

    # Draw the trail duration slider
    draw_slider(screen, 100, HEIGHT + 90, 600, 10, trail_fade_speed / 20, (150, 150, 150), "Trail Duration")

def draw_slider(screen, x, y, width, height, value, color, label):
    # Draw the slider background
    pygame.draw.rect(screen, SLIDER_COLOR, (x, y, width, height))
    # Draw the slider handle
    handle_x = x + value * width
    pygame.draw.circle(screen, color, (int(handle_x), y + height // 2), 8)
    # Render the label
    font = pygame.font.SysFont(None, 24)
    text = font.render(label, True, BLACK)
    screen.blit(text, (x, y - 20))

def handle_slider_event(bodies, mouse_pos, trail_fade_speed):
    for i, body in enumerate(bodies):
        slider_x = 100 + i * 250
        slider_y = HEIGHT + 20
        slider_width = 200
        if slider_x <= mouse_pos[0] <= slider_x + slider_width and slider_y <= mouse_pos[1] <= slider_y + 10:
            relative_position = (mouse_pos[0] - slider_x) / slider_width
            body.mass = relative_position * 1e30

    # Trail duration slider
    slider_x = 100
    slider_y = HEIGHT + 90
    slider_width = 600
    if slider_x <= mouse_pos[0] <= slider_x + slider_width and slider_y <= mouse_pos[1] <= slider_y + 10:
        relative_position = (mouse_pos[0] - slider_x) / slider_width
        trail_fade_speed = max(1, relative_position * 20)  # Scale between 1 and 20
    return trail_fade_speed

def init_bodies():
    M = 1e30
    G = 6.67430e-11
    r = 1e10

    # Positions
    body1_pos = [r, 0]
    body2_pos = [r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3)]
    body3_pos = [r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3)]

    # Velocities
    v = np.sqrt(G * M / r) * 0.75

    angle1 = np.arctan2(body1_pos[1], body1_pos[0]) + np.pi / 2
    angle2 = np.arctan2(body2_pos[1], body2_pos[0]) + np.pi / 2
    angle3 = np.arctan2(body3_pos[1], body3_pos[0]) + np.pi / 2

    body1_vel = [v * np.cos(angle1), v * np.sin(angle1)]
    body2_vel = [v * np.cos(angle2), v * np.sin(angle2)]
    body3_vel = [v * np.cos(angle3), v * np.sin(angle3)]

    body1 = Body(mass=M, position=body1_pos, velocity=body1_vel, color=BODY_COLOR[0])
    body2 = Body(mass=M, position=body2_pos, velocity=body2_vel, color=BODY_COLOR[1])
    body3 = Body(mass=M, position=body3_pos, velocity=body3_vel, color=BODY_COLOR[2])

    return [body1, body2, body3]

def main():
    bodies = init_bodies()
    dt = 1000  # Time step
    offset_x = 0
    offset_y = 0
    is_panning = False
    pan_start_mouse_pos = None
    pan_start_offset = None

    # Create trail surface
    trail_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    clock = pygame.time.Clock()
    running = True
    dragged_body = None

    # Trail fade speed parameter
    trail_fade_speed = 5  # Starting value for trail fade speed

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Mouse button pressed
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                over_body = False
                for body in bodies:
                    if is_mouse_over_body(body, mouse_pos, offset_x, offset_y):
                        dragged_body = body
                        body.is_dragged = True
                        body.prev_mouse_pos = mouse_pos
                        body.mouse_velocity = np.zeros(2)
                        over_body = True
                        print(f"Dragging Body: {body.color}")
                        break

                # Handle slider interaction or panning
                if not over_body:
                    if mouse_pos[1] > HEIGHT:
                        trail_fade_speed = handle_slider_event(bodies, mouse_pos, trail_fade_speed)
                    else:
                        is_panning = True
                        pan_start_mouse_pos = mouse_pos
                        pan_start_offset = (offset_x, offset_y)

            # Mouse button released
            if event.type == pygame.MOUSEBUTTONUP:
                if dragged_body:
                    if clock.get_time() > 0:
                        dragged_body.velocity = (dragged_body.mouse_velocity * scale * mouse_velocity_scale)
                    dragged_body.is_dragged = False
                    dragged_body.prev_mouse_pos = None
                    dragged_body.mouse_velocity = np.zeros(2)
                    dragged_body = None
                    print("Released Body")
                if is_panning:
                    is_panning = False

            # Mouse movement
            if event.type == pygame.MOUSEMOTION:
                if is_panning:
                    mouse_pos = pygame.mouse.get_pos()
                    dx = mouse_pos[0] - pan_start_mouse_pos[0]
                    dy = mouse_pos[1] - pan_start_mouse_pos[1]
                    offset_x = pan_start_offset[0] - dx * scale
                    offset_y = pan_start_offset[1] - dy * scale

        # Update dragged body's position
        if dragged_body:
            mouse_pos = pygame.mouse.get_pos()
            if dragged_body.prev_mouse_pos is not None:
                mouse_displacement = np.array(mouse_pos) - np.array(dragged_body.prev_mouse_pos)
                if clock.get_time() > 0:
                    dragged_body.mouse_velocity = mouse_displacement / clock.get_time()
            dragged_body.position = inverse_transform_position(*mouse_pos, offset_x, offset_y)
            dragged_body.prev_mouse_pos = mouse_pos

        # Update positions and velocities
        update_velocity(bodies, dt)
        update_position(bodies, dt, offset_x, offset_y)

        # Append current position to the positions list for tracing
        for body in bodies:
            # Append current position with full alpha
            body.positions.append({'position': body.position.copy(), 'alpha': 255})

            # Update alpha values
            for trail_point in body.positions:
                trail_point['alpha'] -= trail_fade_speed
            # Remove trail points that are fully transparent
            body.positions = [tp for tp in body.positions if tp['alpha'] > 0]

        # Draw everything
        draw_bodies(bodies, screen, trail_surface, offset_x, offset_y)
        draw_panel(screen, bodies, trail_fade_speed)

        # Control the frame rate
        clock.tick(60)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
