import pygame
import numpy as np

# Initialize Pygame
pygame.init()

G = 6.67430e-11  # Gravitational constant
scale = 1e8      # Scale factor for distances
mouse_velocity_scale = 0.001  # Scale factor for the mouse velocity

# Set up display
WIDTH, HEIGHT = 800, 600
PANEL_HEIGHT = 250  # Increased to accommodate additional info
screen = pygame.display.set_mode((WIDTH, HEIGHT + PANEL_HEIGHT))
pygame.display.set_caption('Three-Body Problem with Boundary Options and Pause')

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

def update_position(bodies, dt, offset_x, offset_y, boundary_behavior):
    for body in bodies:
        if not body.is_dragged:
            body.position += body.velocity * dt

            x_screen, y_screen = transform_position(body.position, offset_x, offset_y)
            RADIUS = 10  # Radius of the bodies

            if boundary_behavior == "Bounce":
                # Bounce behavior
                bounced = False

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

            elif boundary_behavior == "Wrap":
                # Wrapping behavior
                wrapped = False

                if x_screen < -RADIUS:
                    x_screen = WIDTH + RADIUS
                    wrapped = True
                elif x_screen > WIDTH + RADIUS:
                    x_screen = -RADIUS
                    wrapped = True

                if y_screen < -RADIUS:
                    y_screen = HEIGHT + RADIUS
                    wrapped = True
                elif y_screen > HEIGHT + RADIUS:
                    y_screen = -RADIUS
                    wrapped = True

                if wrapped:
                    body.position = inverse_transform_position(x_screen, y_screen, offset_x, offset_y)
            elif boundary_behavior == "None":
                # No interaction with boundaries
                pass

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

def draw_panel(screen, bodies, trail_fade_speed, boundary_behavior, paused):
    # Draw the panel background
    pygame.draw.rect(screen, WHITE, (0, HEIGHT, WIDTH, PANEL_HEIGHT))

    # Set up the font
    font = pygame.font.SysFont(None, 18)
    line_height = font.get_linesize()

    # Define min and max mass exponents for logarithmic scale
    min_mass_exp = 25  # Corresponds to 1e25 kg
    max_mass_exp = 32  # Corresponds to 1e32 kg

    # Draw sliders and display information for each body
    for i, body in enumerate(bodies):
        slider_x = 50 + i * 250
        slider_y = HEIGHT + 20

        # Calculate slider value based on logarithmic mass scale
        mass_exponent = np.log10(body.mass)
        value = (mass_exponent - min_mass_exp) / (max_mass_exp - min_mass_exp)

        draw_slider(screen, slider_x, slider_y, 200, 10, value, BODY_COLOR[i], f"Mass {i+1}")

        # Display mass
        mass_text = font.render(f"Mass: {body.mass:.2e} kg", True, BLACK)
        screen.blit(mass_text, (slider_x, slider_y + 15))

        # Display position
        pos_text = font.render(f"Pos: ({body.position[0]:.2e}, {body.position[1]:.2e}) m", True, BLACK)
        screen.blit(pos_text, (slider_x, slider_y + 15 + line_height))

        # Display velocity
        vel_text = font.render(f"Vel: ({body.velocity[0]:.2e}, {body.velocity[1]:.2e}) m/s", True, BLACK)
        screen.blit(vel_text, (slider_x, slider_y + 15 + 2 * line_height))

        # # Calculate and display the angle of velocity
        # velocity_angle = np.degrees(np.arctan2(body.velocity[1], body.velocity[0]))
        # angle_text = font.render(f"Angle: {velocity_angle:.2f}°", True, BLACK)
        # screen.blit(angle_text, (slider_x, slider_y + 15 + 3 * line_height))

    # Adjusted positions
    # Draw the trail duration slider
    # Calculate slider value based on inverted trail fade speed
    min_trail_fade_speed = 0.2  # Reduced by factor of 5
    max_trail_fade_speed = 4    # Reduced by factor of 5
    value = 1 - ((trail_fade_speed - min_trail_fade_speed) / (max_trail_fade_speed - min_trail_fade_speed))

    trail_slider_y = HEIGHT + 130  # Adjusted position
    trail_slider_width = 400
    draw_slider(screen, 100, trail_slider_y, trail_slider_width, 10, value, (150, 150, 150), "Trail Duration")

    # Draw the boundary behavior button
    button_x = 100 + trail_slider_width + 20  # Place beside the trail slider
    button_y = trail_slider_y - 10  # Adjusted position
    button_width = 120
    button_height = 40
    draw_button(screen, button_x, button_y, button_width, button_height, f"Boundary: {boundary_behavior}")

    # Draw the pause button
    pause_button_x = button_x
    pause_button_y = button_y + button_height + 10  # Slightly below
    pause_button_width = 120
    pause_button_height = 40
    pause_text = "Resume" if paused else "Pause"
    draw_button(screen, pause_button_x, pause_button_y, pause_button_width, pause_button_height, pause_text)

def draw_slider(screen, x, y, width, height, value, color, label):
    # Draw the slider background
    pygame.draw.rect(screen, SLIDER_COLOR, (x, y, width, height))
    # Draw the slider handle
    handle_x = x + value * width
    pygame.draw.circle(screen, color, (int(handle_x), y + height // 2), 8)
    # Render the label
    font = pygame.font.SysFont(None, 18)
    text = font.render(label, True, BLACK)
    screen.blit(text, (x, y - 20))

def draw_button(screen, x, y, width, height, text):
    # Draw the button rectangle
    pygame.draw.rect(screen, SLIDER_COLOR, (x, y, width, height))
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 2)  # Border

    # Render the text
    font = pygame.font.SysFont(None, 18)
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surface, text_rect)

def handle_slider_event(bodies, mouse_pos, trail_fade_speed):
    min_mass_exp = 25
    max_mass_exp = 32

    slider_handled = False  # Flag to indicate if a slider was adjusted

    for i, body in enumerate(bodies):
        slider_x = 50 + i * 250
        slider_y = HEIGHT + 20
        slider_width = 200
        if slider_x <= mouse_pos[0] <= slider_x + slider_width and slider_y <= mouse_pos[1] <= slider_y + 10:
            # Logarithmic mass scaling
            relative_position = (mouse_pos[0] - slider_x) / slider_width
            mass_exponent = min_mass_exp + relative_position * (max_mass_exp - min_mass_exp)
            body.mass = 10 ** mass_exponent
            slider_handled = True

    # Adjusted trail duration slider
    slider_x = 100
    slider_y = HEIGHT + 130  # Adjusted position
    slider_width = 400
    if slider_x <= mouse_pos[0] <= slider_x + slider_width and slider_y <= mouse_pos[1] <= slider_y + 10:
        relative_position = (mouse_pos[0] - slider_x) / slider_width

        # Invert the trail fade speed calculation
        min_trail_fade_speed = 0.2  # Reduced by factor of 5
        max_trail_fade_speed = 4    # Reduced by factor of 5
        trail_fade_speed = min_trail_fade_speed + (1 - relative_position) * (max_trail_fade_speed - min_trail_fade_speed)
        slider_handled = True

    return trail_fade_speed, slider_handled

def handle_button_event(mouse_pos, boundary_behavior, paused):
    # Boundary behavior button
    trail_slider_width = 400
    button_x = 100 + trail_slider_width + 20  # Place beside the trail slider
    button_y = HEIGHT + 130 - 10  # Adjusted position
    button_width = 120
    button_height = 40

    button_handled = False

    if button_x <= mouse_pos[0] <= button_x + button_width and button_y <= mouse_pos[1] <= button_y + button_height:
        # Cycle through boundary behaviors
        if boundary_behavior == "Bounce":
            boundary_behavior = "Wrap"
        elif boundary_behavior == "Wrap":
            boundary_behavior = "None"
        elif boundary_behavior == "None":
            boundary_behavior = "Bounce"
        button_handled = True

    # Pause button
    pause_button_x = button_x
    pause_button_y = button_y + button_height + 10  # Slightly below
    pause_button_width = 120
    pause_button_height = 40

    if pause_button_x <= mouse_pos[0] <= pause_button_x + pause_button_width and pause_button_y <= mouse_pos[1] <= pause_button_y + pause_button_height:
        paused = not paused
        button_handled = True

    return boundary_behavior, paused, button_handled

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

    boundary_behavior = "None"  # Starting with no boundary behavior

    paused = False  # Simulation is running initially

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

                # Handle slider and button interaction or panning
                if not over_body:
                    if mouse_pos[1] > HEIGHT:
                        trail_fade_speed, slider_handled = handle_slider_event(bodies, mouse_pos, trail_fade_speed)
                        boundary_behavior, paused, button_handled = handle_button_event(mouse_pos, boundary_behavior, paused)
                        if not slider_handled and not button_handled:
                            # Start panning if no slider or button was adjusted
                            is_panning = True
                            pan_start_mouse_pos = mouse_pos
                            pan_start_offset = (offset_x, offset_y)
                    else:
                        # Start panning
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

        if not paused:
            # Update positions and velocities
            update_velocity(bodies, dt)
            update_position(bodies, dt, offset_x, offset_y, boundary_behavior)

            # Append current position to the positions list for tracing
            for body in bodies:
                # Append current position with full alpha
                body.positions.append({'position': body.position.copy(), 'alpha': 255})

                # Update alpha values
                for trail_point in body.positions:
                    trail_point['alpha'] -= trail_fade_speed
                # Remove trail points that are fully transparent
                body.positions = [tp for tp in body.positions if tp['alpha'] > 0]
        else:
            # Even when paused, update trails to maintain visual consistency
            for body in bodies:
                # Update alpha values
                for trail_point in body.positions:
                    trail_point['alpha'] -= trail_fade_speed
                # Remove trail points that are fully transparent
                body.positions = [tp for tp in body.positions if tp['alpha'] > 0]

        # Draw everything
        draw_bodies(bodies, screen, trail_surface, offset_x, offset_y)
        draw_panel(screen, bodies, trail_fade_speed, boundary_behavior, paused)

        # Control the frame rate
        clock.tick(60)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
