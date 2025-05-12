import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define theta values
theta0 = 0
theta1 = 1
theta2 = 1

# Create a mesh grid for x1 and x3
x1_vals = np.linspace(0, 5, 50)
x3_vals = np.linspace(0, 5, 50)
x1, x3 = np.meshgrid(x1_vals, x3_vals)

# Calculate x2 from the plane equation
x2 = (-theta0 - theta1 * x1) / theta2

# Plotting the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, x3, alpha=0.7, color='cyan', edgecolor='k')

# Labels and title
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('Plane: θ0 + θ1*x1 + θ2*x2 = 0')

plt.show()