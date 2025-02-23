import numpy as np
import matplotlib.pyplot as plt


def circular_shape(outer_radius, N_pieces, N_rings):
    step_size = 2 * np.pi / N_pieces
    phi = np.linspace(0, 2 * np.pi - step_size, N_pieces,
                      endpoint=True)  # OPEN ring (not fully closed)

    wire_segments = []  # Store wire as a continuous structure
    radius_step = 0.002  # Decrease radius per ring 2mm

    prev_last_point = None  # Track last point of previous ring

    for ring in range(N_rings):
        current_radius = outer_radius - ring * radius_step
        x = current_radius * np.cos(phi)
        y = current_radius * np.sin(phi)
        z = np.zeros_like(x)  # Flat in Z for now

        circle_segment = np.column_stack((x, y, z))  # Open arc
        wire_segments.append(circle_segment)

        # Connect the last point of the previous ring to the first point of the current ring
        if prev_last_point is not None:
            connector_segment = np.array(
                [prev_last_point, circle_segment[0]])  # Transition piece
            wire_segments.append(connector_segment)

        # Update last point tracker for next iteration
        prev_last_point = circle_segment[-1]

    return wire_segments  # Return all segments as one continuous wire


# Call the function
circles = circular_shape(outer_radius=0.12, N_pieces=90, N_rings=20)

# Plot the rings and connections
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Define a single color for the whole wire (e.g., black)
wire_color = 'red'

for segment in circles:
    ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
            color=wire_color, linewidth=1.5)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Final Continuous Wire in Single Color')

plt.show()
