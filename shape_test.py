import numpy as np
import matplotlib.pyplot as plt


def spiral_shape(outer_radius, N_pieces, N_rings):
    step_size = 2 * np.pi / N_pieces
    phi = np.linspace(0, 2 * np.pi - step_size, N_pieces,
                      endpoint=True)  # OPEN ring (not fully closed)

    spiral_segments = []  # Store wire as a continuous structure
    radius_step = 0.002  # Decrease radius per ring 2mm

    prev_last_point = None  # Track last point of previous ring

    for ring in range(N_rings):
        current_radius = outer_radius - ring * radius_step
        x = current_radius * np.cos(phi)
        y = current_radius * np.sin(phi)
        z = np.zeros_like(x)  # Flat in Z for now

        circle_segment = np.column_stack((x, y, z))  # Open arc
        spiral_segments.append(circle_segment)

        # Connect the last point of the previous ring to the first point of the current ring
        if prev_last_point is not None:
            connector_segment = np.array(
                [prev_last_point, circle_segment[0]])  # Transition piece
            spiral_segments.append(connector_segment)

        # Update last point tracker for next iteration
        prev_last_point = circle_segment[-1]

    return spiral_segments  # Return all segments as one continuous wire


def coil_shape(radius, N_pieces, N_rings, z_gap):
    step_size = 2 * np.pi / N_pieces
    phi = np.linspace(0, 2 * np.pi - step_size, N_pieces,
                      endpoint=True)  # OPEN ring (not fully closed)

    coil_segments = []  # Store wire as a continuous structure

    prev_last_point = None  # Track last point of previous ring

    for ring in range(N_rings):
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        z = np.zeros_like(x)+ring*z_gap  # Flat in Z for now

        circle_segment = np.column_stack((x, y, z))  # Open arc
        coil_segments.append(circle_segment)

        # Connect the last point of the previous ring to the first point of the current ring
        if prev_last_point is not None:
            connector_segment = np.array(
                [prev_last_point, circle_segment[0]])  # Transition piece
            coil_segments.append(connector_segment)

        # Update last point tracker for next iteration
        prev_last_point = circle_segment[-1]

    return coil_segments  # Return all segments as one continuous wire


# Call the function
spiral = spiral_shape(outer_radius=0.12, N_pieces=90, N_rings=20)
coil = coil_shape(0.12, 90, 20, 0.001)

# Plot the rings and connections
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(122, projection='3d')
ax2 = fig.add_subplot(121, projection='3d')

# Define a single color for the whole wire (e.g., black)
wire_color = 'red'

for segments in coil:
    ax1.plot(segments[:, 0], segments[:, 1], segments[:, 2],
             color=wire_color, linewidth=1.5)

for segment in spiral:
    ax2.plot(segment[:, 0], segment[:, 1], segment[:, 2],
             color=wire_color, linewidth=1.5)

ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
ax1.set_zlabel('Z [m]')
ax1.set_title('Sprial Wire in Single Color')
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Y [m]')
ax2.set_zlabel('Z [m]')
ax2.set_title('Coil Wire in Single Color')

plt.show()
