import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the surface functions
def surface1(x, y):
    return 2 * x**2 + 2 * y**2

def surface2(x, y):
    return 2 * np.exp(-x**2 - y**2)

# Generate a grid of points in the x-y plane
x = np.linspace(-1, 1, 100)  # Grid for x-axis
y = np.linspace(-1, 1, 100)  # Grid for y-axis
x, y = np.meshgrid(x, y)

# Mask points outside the circular region (x^2 + y^2 <= 1)
mask = x**2 + y**2 <= 1
x = x[mask]
y = y[mask]

# Calculate z values for the two surfaces
z1 = surface1(x, y)  # Top surface
z2 = surface2(x, y)  # Bottom surface

# Define a function to find the plane
def find_balance_plane(z1, z2, x, y):
    z_min = max(z2.min(), z1.min())  # Start at the minimum valid z value
    z_max = min(z2.max(), z1.max())  # End at the maximum valid z value
    tolerance = 1e-3  # Set a tolerance for balance

    # Iterate to find the balanced plane
    for z_plane in np.linspace(z_min, z_max, 1000):  # Fine search over possible z values
        # Count points above and below the plane
        points_above = np.sum(z2 > z_plane)  # Points from surface2 above the plane
        points_below = np.sum(z1 < z_plane)  # Points from surface1 below the plane

        # Check if counts match within tolerance
        if abs(points_above - points_below) <= tolerance:
            return z_plane, points_above, points_below

    return None, None, None  # Return None if no balance found

# Find the z-plane
z_plane, points_above_count, points_below_count = find_balance_plane(z1, z2, x, y)

# Output the results
if z_plane is not None:
    print(f"Balanced Z Plane Found: {z_plane}")
    print(f"Number of Points Above: {points_above_count}")
    print(f"Number of Points Below: {points_below_count}")

    # Save points from surface1 below the plane
    points_below_surface1 = np.c_[x[z1 < z_plane], y[z1 < z_plane], z1[z1 < z_plane]]

    # Save points from surface2 above the plane
    points_above_surface2 = np.c_[x[z2 > z_plane], y[z2 > z_plane], z2[z2 > z_plane]]

    # Combine the two point clouds for visualization
    combined_points = np.vstack([points_below_surface1, points_above_surface2])

    # Plot the point cloud
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot points below the plane from surface1
    ax.scatter(
        points_below_surface1[:, 0], points_below_surface1[:, 1], points_below_surface1[:, 2],
        color="blue", s=1, label="Surface 1 Below Plane"
    )

    # Plot points above the plane from surface2
    ax.scatter(
        points_above_surface2[:, 0], points_above_surface2[:, 1], points_above_surface2[:, 2],
        color="orange", s=1, label="Surface 2 Above Plane"
    )

    # Plot the boundary plane
    xx, yy = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    zz = z_plane * np.ones_like(xx)
    ax.plot_surface(xx, yy, zz, color="green", alpha=0.3, label="Boundary Plane")

    # Set labels and limits
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 4])
    plt.title("Balanced Point Cloud (Surface 1 Below and Surface 2 Above)")
    ax.legend(loc="upper left")

    plt.show()
    plt.savefig("mesh_a.png")
else:
    print("No balanced Z plane found.")
