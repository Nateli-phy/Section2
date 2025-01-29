import numpy as np
import matplotlib.pyplot as plt
import time

# Graham Scan
def graham_scan(points, *args):
    def orientation(p, q, r):
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    points = sorted(points, key=lambda x: (x[0], x[1]))
    lower = []
    for p in points:
        while len(lower) >= 2 and orientation(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and orientation(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])

# Jarvis March
def jarvis_march(points, *args):
    def orientation(p, q, r):
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    n = len(points)
    if n < 3:
        return points
    hull = []
    leftmost = np.argmin(points[:, 0])
    p = leftmost
    while True:
        hull.append(points[p])
        q = (p + 1) % n
        for i in range(n):
            if orientation(points[p], points[i], points[q]) < 0:
                q = i
        p = q
        if p == leftmost:
            break
    return np.array(hull)

# Quickhull
def quickhull(points, *args):
    def find_hull(points, p1, p2):
        if len(points) == 0:
            return []
        distances = np.cross(p2 - p1, points - p1)
        farthest = points[np.argmax(distances)]
        left = points[np.cross(farthest - p1, points - p1) > 0]
        right = points[np.cross(p2 - farthest, points - farthest) > 0]
        return find_hull(left, p1, farthest) + [farthest] + find_hull(right, farthest, p2)
    if len(points) < 3:
        return points
    leftmost = points[np.argmin(points[:, 0])]
    rightmost = points[np.argmax(points[:, 0])]
    upper = points[np.cross(rightmost - leftmost, points - leftmost) > 0]
    lower = points[np.cross(leftmost - rightmost, points - rightmost) > 0]
    hull = [leftmost] + find_hull(upper, leftmost, rightmost) + [rightmost] + find_hull(lower, rightmost, leftmost)
    return np.array(hull)

# Generate uniform point cloud
def generate_uniform_point_cloud(n, seed=None, bounds=(0, 1)):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(bounds[0], bounds[1], size=(n, 2))

# Measure runtime of an algorithm
def measure_runtime(points, algorithm):
    start_time = time.time()
    algorithm(points)
    return time.time() - start_time

# Plot time complexity results
def plot_time_complexity(n_values, results):
    plt.figure(figsize=(10, 6))
    for algo_name, runtimes in results.items():
        plt.plot(n_values, runtimes, label=algo_name)
    plt.xlabel("Number of Points (n)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Time Complexity of Convex Hull Algorithms")
    plt.legend()
    plt.grid()
    plt.savefig("time_complexity_plot1.png")
    plt.show()
    print("Plot saved as 'time_complexity_plot.png'.")

# Main function
def main():
    # Define the convex hull algorithms
    algorithms = {
        "Graham Scan": graham_scan,
        "Jarvis March": jarvis_march,
        "Quickhull": quickhull,
    }

    # Time complexity analysis
    n_values = [10, 50, 100, 200, 400, 800, 1000]
    results = {name: [] for name in algorithms.keys()}

    print("Running time complexity analysis...")
    for n in n_values:
        points = generate_uniform_point_cloud(n, seed=42)
        for name, algo in algorithms.items():
            runtime = measure_runtime(points, algo)
            results[name].append(runtime)

    # Plot and save the results
    plot_time_complexity(n_values, results)

    # Write conclusions to a file
    with open("conclusions.txt", "w") as f:
        f.write("Conclusions:\n")
        f.write("1. Time complexity analysis shows that in general, the time increase as our point number increase, but the rate of increasing is different. \n")
        f.write("2. Differences between algorithms are visible at larger n values. \n")
        f.write("3. Quickhull is often the fastest, while Jarvis March is slower, and the gram scan is also fast compare to Jarvis march, but it is not fast as quick hull.\n")
    print("Conclusions saved in 'conclusions.txt'.")

# Entry point
if __name__ == "__main__":
    main()
