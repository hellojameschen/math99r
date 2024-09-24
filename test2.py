import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Sample path (a circle)
t = np.linspace(0, 2 * np.pi, 10000)
x = t**2
y = t

# Create complex numbers
z = x + 1j * y

# Perform the Fourier transform
fourier_coeffs = np.fft.fft(z)

# Specify how many Fourier coefficients to keep
n_coeffs = 10  # Fixed number of coefficients to be retained

# Create an array of zeros and retain the first 'n_coeffs' coefficients
coeffs = np.zeros_like(fourier_coeffs)
coeffs[:n_coeffs] = fourier_coeffs[:n_coeffs]
coeffs[-n_coeffs + 1:] = fourier_coeffs[-n_coeffs + 1:]

# Reconstruct the path using inverse FFT
z_reconstructed = np.fft.ifft(coeffs)

# Extract x and y coordinates for the reconstructed path
x_reconstructed = np.real(z_reconstructed)
y_reconstructed = np.imag(z_reconstructed)

# Create figure and axis
fig, ax = plt.subplots()
ax.set_xlim(np.min(x_reconstructed), np.max(x_reconstructed))
ax.set_ylim(np.min(y_reconstructed), np.max(y_reconstructed))

# Plot the original path for reference
ax.plot(x, y, label='Original Path', color='gray', alpha=0.5)

# Initialize the line object for the progressively drawn path
line, = ax.plot([], [], label='Reconstructed Path', color='blue')

# Add a legend
ax.legend()

# Initialize the plot with empty data
def init():
    line.set_data([], [])
    return line,

# Update function for the animation to progressively draw the path
def update(frame):
    # Progressively show the reconstructed path up to the current frame
    line.set_data(x_reconstructed[:frame], y_reconstructed[:frame])
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(x_reconstructed), init_func=init, blit=True, interval=1)

# Display the animation
plt.show()
