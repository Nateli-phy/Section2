import numpy as np
import matplotlib.pyplot as plt

# Load the point cloud data
data = np.loadtxt("mesh.dat", delimiter=" ", skiprows=1)

# Extract x and y coordinates
x = data[:, 0]
y = data[:, 1]

# Create a 2D scatter plot
plt.scatter(x, y, c="blue", marker=".")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2D Point Cloud")
plt.show()

# Save the plot to a file
plt.savefig("point_cloud_plot.png")
print("Plot saved as point_cloud_plot.png")
