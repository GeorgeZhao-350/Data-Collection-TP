import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cylinder parameters
radius = 1
height = 5
n_points = 100

# Create a meshgrid for the cylinder
theta = np.linspace(0, 2 * np.pi, n_points)
z = np.linspace(0, height, n_points)
theta, Z = np.meshgrid(theta, z)

# Parametric equations for the cylinder
X = radius * np.cos(theta)+1
Y = radius * np.sin(theta)+1

# Plot the cylinder
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=5, cstride=5, color='b', alpha=0.6)

z_top = height
z_bottom = 0
theta_disc = np.linspace(0, 2 * np.pi, n_points)
X_disc = radius * np.cos(theta_disc)+1
Y_disc = radius * np.sin(theta_disc)+1

# Plot top and bottom circles
ax.plot_trisurf(X_disc, Y_disc, [z_top]*n_points, color='r', alpha=0.6)  # Top surface
ax.plot_trisurf(X_disc, Y_disc, [z_bottom]*n_points, color='r', alpha=0.6)  # Bottom surface

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
