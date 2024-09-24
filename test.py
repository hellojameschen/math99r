import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Sample path (a circle)
t = np.linspace(0, 2 * np.pi, 100000)
x = t**2
y = t

# Create complex numbers
z = x + 1j * y

# Perform the Fourier transform
fourier_coeffs = np.fft.fft(z)

# Number of Fourier coefficients to retain
n_coeffs_max = 200  # Maximum number of coefficients

# Create figure and axis
fig, ax = plt.subplots()
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))

# Plot the original path (static)
ax.plot(x, y, label='Original Path', color='gray', alpha=0.5)

# Initialize the line object for the reconstructed path
line, = ax.plot([], [], label='Reconstructed Path', color='blue')

# Add a legend
ax.legend()

# Initialize function for the animation
def init():
    line.set_data([], [])
    return line,

# Update function for the animation
def update(n_coeffs):
    coeffs = np.zeros_like(fourier_coeffs)  # Create an array of zeros
    # Retain the first 'n_coeffs' Fourier coefficients
    coeffs[:n_coeffs] = fourier_coeffs[:n_coeffs]
    # Retain the corresponding high-frequency coefficients for symmetry
    coeffs[-n_coeffs + 1:] = fourier_coeffs[-n_coeffs + 1:]
    
    # Reconstruct the path using inverse FFT
    z_reconstructed = np.fft.ifft(coeffs)
    
    # Extract x and y coordinates
    x_reconstructed = np.real(z_reconstructed)
    y_reconstructed = np.imag(z_reconstructed)
    
    # Update the line data
    line.set_data(x_reconstructed, y_reconstructed)
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(1, n_coeffs_max), init_func=init, blit=True, interval=50)

# Display the animation
plt.show()
