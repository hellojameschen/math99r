import pygame
import numpy as np

# Initialize Pygame
pygame.init()

BOUNCE = True
# Define the gravitational constant
G = 6.67430e-11
scale = 1e8  # Scale factor for distances
mouse_velocity_scale = 0.001  # Scale factor for the mouse velocity

# Set up display
WIDTH, HEIGHT = 800, 800
PANEL_HEIGHT = 100  # Height of the panel for sliders
screen = pygame.display.set_mode((WIDTH, HEIGHT + PANEL_HEIGHT))  # Add space for panel
pygame.display.set_caption('Three-Body Problem')

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
        self.prev_mouse_pos = None  # Store the previous mouse position for momentum calculation
        self.mouse_velocity = np.zeros(2)  # Store the velocity of the mouse when dragging

def gravitational_force(body1, body2):
    # Calculate the gravitational force between two bodies
    r = np.linalg.norm(body2.position - body1.position)
    force_magnitude = G * body1.mass * body2.mass / r**2
    force_direction = (body2.position - body1.position) / r
    return force_magnitude * force_direction

def update_velocity(bodies, dt):
    # Update velocities based on gravitational forces
    for i, body in enumerate(bodies):
        if not body.is_dragged:  # Only update velocity if the body is not being dragged
            force = np.zeros(2)
            for j, other_body in enumerate(bodies):
                if i != j:
                    force += gravitational_force(body, other_body)
            acceleration = force / body.mass
            body.velocity += acceleration * dt

def update_position(bodies, dt, left_wall_x, right_wall_x, top_wall_y, bottom_wall_y):
    # Update positions based on current velocities and check for wall collisions
    for body in bodies:
        if not body.is_dragged:  # Only update position if the body is not being dragged
            body.position += body.velocity * dt

            if BOUNCE:
                # Left and right walls (x-axis bounce)
                if body.position[0] <= left_wall_x or body.position[0] >= right_wall_x:
                    body.velocity[0] = -body.velocity[0]  # Reverse the x-velocity

                # Top and bottom walls (y-axis bounce)
                if body.position[1] <= top_wall_y or body.position[1] >= bottom_wall_y:
                    body.velocity[1] = -body.velocity[1]  # Reverse the y-velocity

def draw_bodies(bodies, screen, offset_x, offset_y):
    # Draw the bodies on the screen
    screen.fill(BLACK, (0, 0, WIDTH, HEIGHT))  # Clear the simulation area
    for body in bodies:
        x, y = transform_position(body.position, offset_x, offset_y)
        pygame.draw.circle(screen, body.color, (int(x), int(y)), 10)  # Draw the body as a small circle

def transform_position(position, offset_x, offset_y):
    # Transform position to fit on screen
    x = WIDTH // 2 + (position[0] - offset_x) / scale
    y = HEIGHT // 2 + (position[1] - offset_y) / scale
    return x, y

def inverse_transform_position(x, y, offset_x, offset_y):
    # Convert screen coordinates back to physical space
    pos_x = (x - WIDTH // 2) * scale + offset_x
    pos_y = (y - HEIGHT // 2) * scale + offset_y
    return np.array([pos_x, pos_y])

def is_mouse_over_body(body, mouse_pos, offset_x, offset_y):
    # Check if the mouse is over the body
    x, y = transform_position(body.position, offset_x, offset_y)
    distance = np.linalg.norm(np.array(mouse_pos) - np.array([x, y]))
    return distance < 10  # 10 is the radius of the circle

def draw_panel(screen, bodies):
    # Draw the panel at the bottom of the screen
    pygame.draw.rect(screen, WHITE, (0, HEIGHT, WIDTH, PANEL_HEIGHT))  # Background of the panel

    # Draw sliders for each body's mass
    for i, body in enumerate(bodies):
        draw_slider(screen, 100 + i * 250, HEIGHT + 50, 200, 10, body.mass / 1e30, BODY_COLOR[i])

def draw_slider(screen, x, y, width, height, value, color):
    # Draw the slider background
    pygame.draw.rect(screen, SLIDER_COLOR, (x, y, width, height))
    # Draw the slider handle
    handle_x = x + value * width
    pygame.draw.circle(screen, color, (int(handle_x), y + height // 2), 8)

def handle_slider_event(bodies, mouse_pos):
    # Check if the user is interacting with any of the sliders
    for i, body in enumerate(bodies):
        slider_x = 100 + i * 250
        slider_y = HEIGHT + 50
        slider_width = 200
        if slider_x <= mouse_pos[0] <= slider_x + slider_width and slider_y <= mouse_pos[1] <= slider_y + 10:
            # Calculate new mass based on the slider position
            relative_position = (mouse_pos[0] - slider_x) / slider_width
            body.mass = relative_position * 1e30  # Scale mass between 0 and 1e30

def init_bodies():
    # Mass and gravitational constant
    M = 1e30  # Mass of each body
    G = 6.67430e-11  # Gravitational constant
    r = 1e10

    # Positions of the bodies
    body1_pos = [r, 0]
    body2_pos = [r * np.cos(2*np.pi/3), r * np.sin(2*np.pi/3)]
    body3_pos = [r * np.cos(4*np.pi/3), r * np.sin(4*np.pi/3)]


    # Calculate the required velocity for a stable orbit
    v = np.sqrt(G * M / r)*0.75

    # Calculate the angle of the radius vector for each body
    angle1 = np.arctan2(body1_pos[1], body1_pos[0]) + np.pi / 2
    angle2 = np.arctan2(body2_pos[1], body2_pos[0]) + np.pi / 2
    angle3 = np.arctan2(body3_pos[1], body3_pos[0]) + np.pi / 2

    # Set the velocities perpendicular to the radius vectors
    body1_vel = [v * np.cos(angle1), v * np.sin(angle1)]
    body2_vel = [v * np.cos(angle2), v * np.sin(angle2)]
    body3_vel = [v * np.cos(angle3), v * np.sin(angle3)]

    # Create bodies
    body1 = Body(mass=M, position=body1_pos, velocity=body1_vel, color=BODY_COLOR[0])
    body2 = Body(mass=M, position=body2_pos, velocity=body2_vel, color=BODY_COLOR[1])
    body3 = Body(mass=M, position=body3_pos, velocity=body3_vel, color=BODY_COLOR[2])
    return [body1, body2, body3]

def main():
    # Create bodies
    bodies = init_bodies()

    # Simulation parameters
    dt = 1000  # Adjusted time step

    # Initialize variables for panning
    offset_x = 0
    offset_y = 0
    is_panning = False
    pan_start_mouse_pos = None
    pan_start_offset = None

    # Pygame main loop
    clock = pygame.time.Clock()
    running = True
    dragged_body = None  # To keep track of the body being dragged

    while running:
        # Check for quit event
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
                        body.prev_mouse_pos = mouse_pos  # Store the initial mouse position
                        body.mouse_velocity = np.zeros(2)  # Reset mouse velocity when starting drag
                        over_body = True
                        print(f"Dragging Body: {body.color}")
                        break  # Only drag one body

                # Handle slider interaction
                if not over_body:
                    if mouse_pos[1] > HEIGHT:
                        handle_slider_event(bodies, mouse_pos)
                    else:
                        # If not over a body or slider, start panning
                        is_panning = True
                        pan_start_mouse_pos = mouse_pos
                        pan_start_offset = (offset_x, offset_y)

            # Mouse button released
            if event.type == pygame.MOUSEBUTTONUP:
                if dragged_body:
                    # Apply the last calculated mouse velocity to the body with scaling
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

        # If dragging a body, update its position to follow the mouse and track velocity
        if dragged_body:
            mouse_pos = pygame.mouse.get_pos()
            if dragged_body.prev_mouse_pos is not None:
                # Calculate the velocity of the mouse in screen coordinates
                mouse_displacement = np.array(mouse_pos) - np.array(dragged_body.prev_mouse_pos)
                if clock.get_time() > 0:
                    dragged_body.mouse_velocity = mouse_displacement / (clock.get_time())  # Pixels per millisecond
            dragged_body.position = inverse_transform_position(*mouse_pos, offset_x, offset_y)
            dragged_body.prev_mouse_pos = mouse_pos  # Update previous mouse position

        # Calculate the physical positions of the walls after panning
        left_wall_x = offset_x - WIDTH // 2 * scale
        right_wall_x = offset_x + WIDTH // 2 * scale
        top_wall_y = offset_y - HEIGHT // 2 * scale
        bottom_wall_y = offset_y + HEIGHT // 2 * scale

        # Update positions and velocities
        update_velocity(bodies, dt)
        update_position(bodies, dt, left_wall_x, right_wall_x, top_wall_y, bottom_wall_y)

        # Draw the current state
        draw_bodies(bodies, screen, offset_x, offset_y)
        draw_panel(screen, bodies)  # Draw the panel with sliders

        # Control the frame rate
        clock.tick(60)  # Limit to 60 FPS

        pygame.display.flip()

    # Quit Pygame
    pygame.quit()

if __name__ == "__main__":
    main()
