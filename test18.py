import pygame
import numpy as np

# Initialize Pygame
pygame.init()

G = 6.67430e-11  # Gravitational constant
scale = 1e8      # Scale factor for distances
mouse_velocity_scale = 0.001  # Scale factor for the mouse velocity

# Define min and max scale for zooming
MIN_SCALE = 1e6
MAX_SCALE = 1e10
LOG_MIN_SCALE = np.log10(MIN_SCALE)
LOG_MAX_SCALE = np.log10(MAX_SCALE)

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

class InputBox:
    def __init__(self, x, y, w, h, text='', font_size=18):
        self.rect = pygame.Rect(x, y, w, h)
        self.color_inactive = SLIDER_COLOR
        self.color_active = (255, 255, 255)
        self.color = self.color_inactive
        self.text = text
        self.font = pygame.font.SysFont(None, font_size)
        self.txt_surface = self.font.render(text, True, BLACK)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect
            if self.rect.collidepoint(event.pos):
                # Toggle active state
                self.active = True
            else:
                self.active = False
            # Change the color of the input box
            self.color = self.color_active if self.active else self.color_inactive
        elif event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    # Deactivate the input box
                    self.active = False
                    self.color = self.color_inactive
                    # Return the text
                    return self.text
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text
                self.txt_surface = self.font.render(self.text, True, BLACK)
        return None

    def draw(self, screen):
        # Blit the text
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect
        pygame.draw.rect(screen, self.color, self.rect, 2)

class Dropdown:
    def __init__(self, x, y, width, height, options, selected_index=0):
        self.rect = pygame.Rect(x, y, width, height)
        self.options = options
        self.selected_index = selected_index
        self.is_open = False
        self.font = pygame.font.SysFont(None, 18)
        self.bg_color = SLIDER_COLOR
        self.text_color = BLACK
        self.option_rects = []

    def draw(self, screen):
        # Draw the dropdown box
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)  # Border
        # Render the selected option
        selected_text = self.font.render(self.options[self.selected_index], True, self.text_color)
        text_rect = selected_text.get_rect()
        text_rect.centery = self.rect.centery
        text_rect.x = self.rect.x + 5  # Padding
        screen.blit(selected_text, text_rect)

        # Draw the dropdown arrow
        arrow = "▼" if not self.is_open else "▲"
        arrow_text = self.font.render(arrow, True, self.text_color)
        arrow_rect = arrow_text.get_rect()
        arrow_rect.centery = self.rect.centery
        arrow_rect.right = self.rect.right - 5  # Padding
        screen.blit(arrow_text, arrow_rect)

        # If open, draw the options
        if self.is_open:
            # Background for options
            options_height = len(self.options) * self.rect.height
            options_rect = pygame.Rect(self.rect.x, self.rect.y + self.rect.height, self.rect.width, options_height)
            pygame.draw.rect(screen, self.bg_color, options_rect)
            pygame.draw.rect(screen, BLACK, options_rect, 2)
            self.option_rects = []
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + self.rect.height * (i + 1), self.rect.width, self.rect.height)
                self.option_rects.append(option_rect)
                pygame.draw.rect(screen, self.bg_color, option_rect)
                pygame.draw.rect(screen, BLACK, option_rect, 1)
                option_text = self.font.render(option, True, self.text_color)
                text_rect = option_text.get_rect()
                text_rect.centery = option_rect.centery
                text_rect.x = option_rect.x + 5  # Padding
                screen.blit(option_text, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            if self.rect.collidepoint(mouse_pos):
                # Toggle dropdown
                self.is_open = not self.is_open
                return None  # No selection change
            elif self.is_open:
                # Check if an option is clicked
                for i, option_rect in enumerate(self.option_rects):
                    if option_rect.collidepoint(mouse_pos):
                        self.selected_index = i
                        self.is_open = False
                        return self.options[i]  # Return the selected option
                # Clicked outside, close the dropdown
                self.is_open = False
        return None

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

def draw_panel(screen, bodies, trail_fade_speed, paused, boundary_dropdown, input_boxes, scale):
    # Draw the panel background
    pygame.draw.rect(screen, WHITE, (0, HEIGHT, WIDTH, PANEL_HEIGHT))

    # Set up the font
    font = pygame.font.SysFont(None, 18)
    line_height = font.get_linesize()

    # Define min and max mass exponents for logarithmic scale
    min_mass_exp = 25  # Corresponds to 1e25 kg
    max_mass_exp = 32  # Corresponds to 1e32 kg

    # Draw sliders and display information for each body
    for i, (body, input_box_pair) in enumerate(zip(bodies, input_boxes)):
        slider_x = 50 + i * 250
        slider_y = HEIGHT + 20

        # Calculate slider value based on logarithmic mass scale
        mass_exponent = np.log10(body.mass)
        value = (mass_exponent - min_mass_exp) / (max_mass_exp - min_mass_exp)

        draw_slider(screen, slider_x, slider_y, 200, 10, value, BODY_COLOR[i], f"Mass {i+1}")

        # Display mass
        mass_text = font.render(f"Mass: {body.mass:.2e} kg", True, BLACK)
        screen.blit(mass_text, (slider_x, slider_y + 15))

        # Draw position
        if paused:
            # Update InputBox text if not active
            pos_input_box = input_box_pair['pos']
            if not pos_input_box.active:
                pos_input_box.text = f"{body.position[0]:.2e}, {body.position[1]:.2e}"
                pos_input_box.txt_surface = pos_input_box.font.render(pos_input_box.text, True, BLACK)
            pos_input_box.rect.topleft = (slider_x, slider_y + 15 + line_height)
            pos_input_box.draw(screen)
        else:
            # Display position
            pos_text = font.render(f"Pos: ({body.position[0]:.2e}, {body.position[1]:.2e}) m", True, BLACK)
            screen.blit(pos_text, (slider_x, slider_y + 15 + line_height))

        # Draw velocity
        if paused:
            # Update InputBox text if not active
            vel_input_box = input_box_pair['vel']
            if not vel_input_box.active:
                vel_input_box.text = f"{body.velocity[0]:.2e}, {body.velocity[1]:.2e}"
                vel_input_box.txt_surface = vel_input_box.font.render(vel_input_box.text, True, BLACK)
            vel_input_box.rect.topleft = (slider_x, slider_y + 15 + 2 * line_height)
            vel_input_box.draw(screen)
        else:
            # Display velocity
            vel_text = font.render(f"Vel: ({body.velocity[0]:.2e}, {body.velocity[1]:.2e}) m/s", True, BLACK)
            screen.blit(vel_text, (slider_x, slider_y + 15 + 2 * line_height))

    # Adjusted positions
    # Calculate slider value for trail duration
    if trail_fade_speed is None:
        value = 0  # No trail
    else:
        max_trail_fade_speed = 4
        value = 1 - (trail_fade_speed / max_trail_fade_speed)

    trail_slider_y = HEIGHT + 130  # Adjusted position
    trail_slider_width = 400
    draw_slider(screen, 100, trail_slider_y, trail_slider_width, 10, value, (150, 150, 150), "Trail Duration")

    # Zoom slider
    # Calculate slider value based on logarithmic scale
    log_scale = np.log10(scale)
    zoom_value = (log_scale - LOG_MIN_SCALE) / (LOG_MAX_SCALE - LOG_MIN_SCALE)

    zoom_slider_y = trail_slider_y + 50  # Adjusted position
    zoom_slider_width = 400
    draw_slider(screen, 100, zoom_slider_y, zoom_slider_width, 10, zoom_value, (150, 150, 150), "Zoom Level")

    # Draw the boundary behavior dropdown
    boundary_dropdown.draw(screen)

    # Draw the pause button
    pause_button_x = boundary_dropdown.rect.x
    pause_button_y = boundary_dropdown.rect.y + boundary_dropdown.rect.height + 10  # Slightly below
    pause_button_width = boundary_dropdown.rect.width
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

def handle_slider_event(bodies, mouse_pos, trail_fade_speed, scale):
    min_mass_exp = 25
    max_mass_exp = 32
    global LOG_MIN_SCALE, LOG_MAX_SCALE

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
    trail_slider_y = HEIGHT + 130
    slider_y = trail_slider_y  # Adjusted position
    slider_width = 400
    if slider_x <= mouse_pos[0] <= slider_x + slider_width and slider_y <= mouse_pos[1] <= slider_y + 10:
        relative_position = (mouse_pos[0] - slider_x) / slider_width

        if relative_position <= 0.01:
            trail_fade_speed = None  # No trail
        elif relative_position >= 0.99:
            trail_fade_speed = 0  # Infinite trail
        else:
            max_trail_fade_speed = 4
            trail_fade_speed = max_trail_fade_speed * (1 - relative_position)

        slider_handled = True

    # Zoom slider
    slider_x = 100
    zoom_slider_y = trail_slider_y + 50
    slider_y = zoom_slider_y  # Should match the y-position in draw_panel
    slider_width = 400
    if slider_x <= mouse_pos[0] <= slider_x + slider_width and slider_y <= mouse_pos[1] <= slider_y + 10:
        relative_position = (mouse_pos[0] - slider_x) / slider_width
        # Clamp relative_position to [0,1]
        relative_position = max(0, min(relative_position, 1))
        log_scale = LOG_MIN_SCALE + relative_position * (LOG_MAX_SCALE - LOG_MIN_SCALE)
        scale = 10 ** log_scale
        slider_handled = True

    return trail_fade_speed, slider_handled, scale

def handle_pause_button_event(mouse_pos, paused, boundary_dropdown):
    # Pause button
    pause_button_x = boundary_dropdown.rect.x
    pause_button_y = boundary_dropdown.rect.y + boundary_dropdown.rect.height + 10  # Slightly below
    pause_button_width = boundary_dropdown.rect.width
    pause_button_height = 40

    if pause_button_x <= mouse_pos[0] <= pause_button_x + pause_button_width and pause_button_y <= mouse_pos[1] <= pause_button_y + pause_button_height:
        paused = not paused
        return paused, True
    return paused, False

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
    global scale
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
    # Initialize trail_fade_speed to correspond to infinite trail duration
    trail_fade_speed = 1e10  # no trail

    boundary_behavior = "None"  # Starting with no boundary behavior

    paused = False  # Simulation is running initially

    # Create boundary behavior dropdown
    trail_slider_width = 400
    dropdown_x = 100 + trail_slider_width + 20  # Place beside the trail slider
    dropdown_y = HEIGHT + 130 - 10  # Adjusted position
    dropdown_width = 120
    dropdown_height = 40
    boundary_options = ["None", "Wrap", "Bounce"]
    boundary_dropdown = Dropdown(dropdown_x, dropdown_y, dropdown_width, dropdown_height, boundary_options, selected_index=boundary_options.index(boundary_behavior))

    # Initialize InputBoxes for position and velocity
    input_boxes = []
    for i, body in enumerate(bodies):
        slider_x = 50 + i * 250
        slider_y = HEIGHT + 20
        line_height = 18  # or use font.get_linesize()
        font_size = 18

        # Position InputBox
        pos_input_box = InputBox(
            slider_x, slider_y + 15 + line_height, 200, line_height + 10,
            text=f"{body.position[0]:.2e}, {body.position[1]:.2e}",
            font_size=font_size
        )
        # Velocity InputBox
        vel_input_box = InputBox(
            slider_x, slider_y + 15 + 2 * line_height, 200, line_height + 10,
            text=f"{body.velocity[0]:.2e}, {body.velocity[1]:.2e}",
            font_size=font_size
        )
        input_boxes.append({'pos': pos_input_box, 'vel': vel_input_box})

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
                        trail_fade_speed, slider_handled, scale = handle_slider_event(bodies, mouse_pos, trail_fade_speed, scale)
                        paused, pause_button_handled = handle_pause_button_event(mouse_pos, paused, boundary_dropdown)
                        # Pass event to dropdown
                        boundary_selection = boundary_dropdown.handle_event(event)
                        if boundary_selection is not None:
                            boundary_behavior = boundary_selection
                        if not slider_handled and not pause_button_handled and boundary_selection is None:
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
                # Pass event to dropdown
                boundary_dropdown.handle_event(event)

            # Mouse movement
            if event.type == pygame.MOUSEMOTION:
                if is_panning:
                    mouse_pos = pygame.mouse.get_pos()
                    dx = mouse_pos[0] - pan_start_mouse_pos[0]
                    dy = mouse_pos[1] - pan_start_mouse_pos[1]
                    offset_x = pan_start_offset[0] - dx * scale
                    offset_y = pan_start_offset[1] - dy * scale
                # Pass event to dropdown
                boundary_dropdown.handle_event(event)

            # Handle events for InputBoxes when paused
            if paused:
                for i, input_box_pair in enumerate(input_boxes):
                    for key in ['pos', 'vel']:
                        input_box = input_box_pair[key]
                        result = input_box.handle_event(event)
                        if result is not None:
                            # User has finished editing (e.g., pressed Enter)
                            body = bodies[i]
                            text = input_box.text
                            # Parse the text into numbers
                            try:
                                values = [float(val.strip()) for val in text.split(',')]
                                if len(values) == 2:
                                    if key == 'pos':
                                        body.position = np.array(values, dtype=np.float64)
                                        body.positions = []
                                    elif key == 'vel':
                                        body.velocity = np.array(values, dtype=np.float64)
                                else:
                                    print("Please enter two comma-separated numbers.")
                            except ValueError:
                                print("Invalid input. Please enter numeric values.")

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
            # Deactivate all InputBoxes when not paused
            for input_box_pair in input_boxes:
                for key in ['pos', 'vel']:
                    input_box = input_box_pair[key]
                    input_box.active = False
                    input_box.color = input_box.color_inactive

            # Update positions and velocities
            update_velocity(bodies, dt)
            update_position(bodies, dt, offset_x, offset_y, boundary_behavior)

            for body in bodies:
                if trail_fade_speed is not None:
                    # Append current position with full alpha
                    body.positions.append({'position': body.position.copy(), 'alpha': 255})

                    # Update alpha values
                    if trail_fade_speed > 0:
                        for trail_point in body.positions:
                            trail_point['alpha'] -= trail_fade_speed
                        # Remove trail points that are fully transparent
                        body.positions = [tp for tp in body.positions if tp['alpha'] > 0]
                else:
                    # Clear trail points when trail_fade_speed is None
                    body.positions = []
        else:
            # Even when paused, update trails
            for body in bodies:
                if trail_fade_speed is not None and trail_fade_speed > 0:
                    for trail_point in body.positions:
                        trail_point['alpha'] -= trail_fade_speed
                    body.positions = [tp for tp in body.positions if tp['alpha'] > 0]
                elif trail_fade_speed is None:
                    body.positions = []

        # Draw everything
        draw_bodies(bodies, screen, trail_surface, offset_x, offset_y)
        draw_panel(screen, bodies, trail_fade_speed, paused, boundary_dropdown, input_boxes, scale)

        # Control the frame rate
        clock.tick(60)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
