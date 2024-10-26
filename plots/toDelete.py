# Updated code to label the corner points on the plot
import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(0, 10, 400)

# Corresponding y values for the inequalities
y1 = (48 - 8 * x) / 6  # 6y + 8x ≤ 48 rearranged as y ≤ (48 - 8x) / 6
y2 = 6  # y ≤ 6
y3 = 0  # y ≥ 0, implicit from the constraints

# Set up the plot
plt.figure(figsize=(8, 6))

# Plot the boundary lines
plt.plot(x, y1, label=r'$6y + 8x \leq 48$', color='red')
plt.axhline(6, label=r'$y \leq 6$', color='green')  # y ≤ 6 as a horizontal line
plt.axvline(4, label=r'$x \leq 4$', color='blue')  # x ≤ 4 as a vertical line

# Shading the feasible region
plt.fill_between(x, np.minimum(y1, 6), where=(x <= 4), color='gray', alpha=0.3)

# Define the corner points
corner_points = [(0, 6), (2, 6), (4, 3), (4, 0)]

# Plot and label the corner points
for point in corner_points:
    plt.plot(point[0], point[1], 'ro')  # Red dots at corner points
    plt.text(point[0] + 0.2, point[1] - 0.3, f'({point[0]}, {point[1]})', color='black')

# Set limits and labels
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Feasible Region with Labeled Corner Points')

# Display legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
