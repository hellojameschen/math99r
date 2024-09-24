import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Three-Body Problem')

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BODY_COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue

# Define the gravitational constant
G = 6.67430e-11
scale = 1e9

class Body:
    def __init__(self, mass, position, velocity, color):
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.color = color
        self.is_dragged = False

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

def update_position(bodies, dt):
    # Update positions based on current velocities
    for body in bodies:
        if not body.is_dragged:  # Only update position if the body is not being dragged
            body.position += body.velocity * dt

def draw_bodies(bodies, screen):
    # Draw the bodies on the screen
    screen.fill(BLACK)  # Clear the screen
    for body in bodies:
        x, y = transform_position(body.position)
        pygame.draw.circle(screen, body.color, (int(x), int(y)), 10)  # Draw the body as a small circle
    pygame.display.flip()

def transform_position(position):
    # Transform position to fit on screen
    x = WIDTH // 2 + position[0] / scale
    y = HEIGHT // 2 + position[1] / scale
    return x, y

def inverse_transform_position(x, y):
    # Convert screen coordinates back to physical space
    pos_x = (x - WIDTH // 2) * scale
    pos_y = (y - HEIGHT // 2) * scale
    return np.array([pos_x, pos_y])

def is_mouse_over_body(body, mouse_pos):
    # Check if the mouse is over the body
    x, y = transform_position(body.position)
    distance = np.linalg.norm(np.array(mouse_pos) - np.array([x, y]))
    return distance < 10  # 10 is the radius of the circle

def main():
    # Create bodies
    body1 = Body(mass=1e30, position=[-1e11, 0], velocity=[0, 1e4], color=BODY_COLOR[0])
    body2 = Body(mass=1e30, position=[1e11, 0], velocity=[0, -1e4], color=BODY_COLOR[1])
    body3 = Body(mass=1e30, position=[0, 1e11], velocity=[-1e4, 0], color=BODY_COLOR[2])

    bodies = [body1, body2, body3]

    # Simulation parameters
    dt = 10000  # Time step in seconds

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
                for body in bodies:
                    if is_mouse_over_body(body, mouse_pos):
                        dragged_body = body
                        body.is_dragged = True
                        print(f"Dragging Body: {body.color}")

            # Mouse button released
            if event.type == pygame.MOUSEBUTTONUP:
                if dragged_body:
                    dragged_body.is_dragged = False
                    dragged_body = None
                    print("Released Body")

        # If dragging a body, update its position to follow the mouse
        if dragged_body:
            mouse_pos = pygame.mouse.get_pos()
            dragged_body.position = inverse_transform_position(*mouse_pos)

        # Update positions and velocities
        update_velocity(bodies, dt)
        update_position(bodies, dt)

        # Draw the current state
        draw_bodies(bodies, screen)

        # Control the frame rate
        clock.tick(60)  # Limit to 60 FPS

    # Quit Pygame
    pygame.quit()

if __name__ == "__main__":
    main()
