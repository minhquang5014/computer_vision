import matplotlib.pyplot as plt
import math

# Points
x1, y1 = 3, 4
x2, y2 = 6, 8

# Calculate differences
dx = x2 - x1
dy = y2 - y1

# Calculate Euclidean distance using math.hypot
distance = math.hypot(dx, dy)

# Plot points A and B
plt.scatter([x1, x2], [y1, y2], color='blue', label='Points A and B')
plt.annotate('A', (x1, y1), textcoords="offset points", xytext=(5,-5), ha='center')
plt.annotate('B', (x2, y2), textcoords="offset points", xytext=(5,-5), ha='center')

# Plot the line connecting points A and B
plt.plot([x1, x2], [y1, y2], color='red', linestyle='-', linewidth=1, label='Distance')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Euclidean Distance between Points A and B')

# Display the Euclidean distance
plt.text((x1 + x2) / 2, (y1 + y2) / 2, f'Distance = {distance:.2f}', fontsize=12, ha='center')

# Add legend
plt.legend()

# Set equal scaling for better visualization
plt.gca().set_aspect('equal', adjustable='box')

# Show plot
plt.grid(True)
plt.show()