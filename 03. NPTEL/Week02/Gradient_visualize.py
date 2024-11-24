import numpy as np
import matplotlib.pyplot as plt

# Input data
X = [0.5, 2.5]
Y = [0.2, 0.9]

# Sigmoid function
def f(w, b, x):
    return 1.0 / (1.0 + np.exp(-(w * x + b)))

# Error function
def error(w, b):
    err = 0.0
    for x, y in zip(X, Y):
        fx = f(w, b, x)
        err += 0.5 * (fx - y) ** 2
    return err

# Generate a grid of w and b values
w_values = np.linspace(-4, 4, 100)  # Weight values from -4 to 4
b_values = np.linspace(-4, 4, 100)  # Bias values from -4 to 4
W, B = np.meshgrid(w_values, b_values)  # Create a grid of (w, b) pairs

# Compute the error for each (w, b) pair
Error = np.zeros_like(W)  # Initialize an array for error values
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Error[i, j] = error(W[i, j], B[i, j])  # Calculate error for each pair

# Plot the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
surf = ax.plot_surface(W, B, Error, cmap='viridis', edgecolor='none')
ax.set_title("Error Surface")
ax.set_xlabel("Weight (w)")
ax.set_ylabel("Bias (b)")
ax.set_zlabel("Error")

# Add a color bar to show the error magnitude
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.show()
