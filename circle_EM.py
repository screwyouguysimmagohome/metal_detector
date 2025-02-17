# Dette er en python fil, hvori jeg vil forsøge at modellere EM feldt omkring en circulær ledning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import grid3DFunc as grid3d
import scipy.integrate as integrate


mu0 = 4*np.pi*1e-7  # vaccum permeability


def define_circle(radius, definition):
    """
    Generate an array of points along the circumference of a circle in the XY plane.

    Parameters:
    - radius (float): The radius of the circle.
    - definition (int): The number of points to generate along the circle.

    Returns:
    - numpy.ndarray: An array of shape (definition, 3) where each row represents 
      a point (x, y, z) on the circle. The circle lies in the XY plane (z=0 for all points).
    """
    step_size = 360/definition
    phi = np.arange(0+step_size, 360+step_size, step_size)*np.pi/180
    x_angles = np.cos(phi)*radius
    y_angles = np.sin(phi)*radius
    z_angles = np.zeros(x_angles.shape)*radius
    circle = np.column_stack((x_angles, y_angles, z_angles))

    return circle

# circle = circle_radius*(np.cos(phi*x_vec)+np.sin(phi*y_vec))

# print(x_angles.shape)
# print(y_angles.shape)
# print(z_angles.shape)
# print(phi.shape)
# print(circle.shape)
# print(circle)


'''
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(circle[:, 0], circle[:, 1], circle[:, 2],
        color='r', label='circle points')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("wire circle")
ax.legend()

plt.show()'''


def biot_savart_integral(wire_points, field_points, I):
    wire_points = np.asarray(wire_points)
    field_points = np.asarray(field_points)

    # Remember the original shape (in case it's 4D or more)
    orig_shape = field_points.shape

    # If the last dimension is 3 but there are more than 2 total dimensions
    # (e.g. (nx, ny, nz, 3)), flatten to (M, 3)
    if field_points.ndim > 2 and field_points.shape[-1] == 3:
        field_points = field_points.reshape(-1, 3)

    # Build dL and segment midpoints
    dl = wire_points[1:] - wire_points[:-1]
    r_seg = 0.5*(wire_points[1:] + wire_points[:-1])

    # Initialize B_total for the flattened field points
    B_total = np.zeros((field_points.shape[0], 3), dtype=float)

    # Accumulate contributions from each wire segment
    for dLi, seg_pos in zip(dl, r_seg):
        r_vec = field_points - seg_pos  # shape: (M, 3)
        r_mag = np.linalg.norm(r_vec, axis=1, keepdims=True)
        r_mag[r_mag < 1e-14] = 1e-14   # avoid singularities

        cross_term = np.cross(dLi, r_vec)
        B_total += cross_term / (r_mag**3)

    # Apply the Biot–Savart prefactor
    B_total *= mu0 * I / (4*np.pi)

    # Reshape back if we flattened
    if field_points.shape[0] != orig_shape[0]:  # means we did flatten
        # Restore to the original shape, e.g. (nx, ny, nz, 3)
        B_total = B_total.reshape(*orig_shape[:-1], 3)

    return B_total


def EM_of_circle():
    circle = define_circle(0.05, 360)
    grid = grid3d.grid_3D(-0.1, 0.1, 0.01, -0.1, 0.1, 0.01, -0.1, 0.1, 0.01)
    B = biot_savart_integral(circle, grid, 10)
    print(B)


EM_of_circle()
