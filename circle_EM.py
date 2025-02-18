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

    # Build dL and segment midpoints
    dl = wire_points[1:, :] - wire_points[:-1, :]
    r_seg = 0.5*(wire_points[1:, :] + wire_points[:-1, :])

    # Initialize B_total for the flattened field points
    B_total = np.zeros((field_points.shape), dtype=float)

    # Accumulate contributions from each wire segment
    for dLi, seg_pos in zip(dl, r_seg):
        r_vec = field_points - seg_pos  # shape: (21, 21, 21, 3)
        # shape: (21,1,21,3)
        r_mag = np.linalg.norm(r_vec, axis=-1, keepdims=True)
        # r_mag=np.reshape(r_mag,(21,21,21,3))
        r_mag[r_mag < 1e-14] = 1e-14   # avoid singularities

        cross_term = np.cross(dLi, r_vec)
        # breakpoint()
        B_total += cross_term / r_mag

    # Apply the Biot–Savart prefactor
    B_total *= mu0 * I / (4*np.pi)

    return B_total


'''
def EM_of_circle():
    circle = define_circle(0.05, 45)
    grid = grid3d.grid_3D(-0.1, 0.1, 0.01, -0.1, 0.1, 0.01, -0.1, 0.1, 0.01)
    B = biot_savart_integral(circle, grid, 10)
   # print(B)
'''


circle = define_circle(0.05, 45)


def plot_magnetic_field(grid, B_field):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Brug hele griddet
    X = grid[..., 0]
    Y = grid[..., 1]
    Z = grid[..., 2]

    U = B_field[..., 0]  # Magnetfelt i x-retning
    V = B_field[..., 1]  # Magnetfelt i y-retning
    W = B_field[..., 2]  # Magnetfelt i z-retning

    # Beregn magnetfeltets størrelse
    B_magnitude = np.sqrt(U**2 + V**2 + W**2)
    B_magnitude[B_magnitude == 0] = np.min(B_magnitude[B_magnitude > 0])

    print("Max B:", np.max(B_magnitude))
    print("Min B:", np.min(B_magnitude))

    # Brug quiver til at tegne vektorer, hvor længden svarer til magnetfeltets styrke
    ax.quiver(X, Y, Z, U*1e4, V*1e4, W*1e4,
              length=1.0, normalize=False, color='r')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Magnetfeltets retning og styrke i 3D")

    plt.show()


def Emfield_plot_behavior_circle_wire():
    grid = grid3d.grid_3D(
        -0.05, 0.05, 0.02, -0.05, 0.05, 0.02, -0.08, 0.08, 0.02)

    B_field = np.zeros(grid.shape[:3]+(3,))

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                B_field[i, j, k] = biot_savart_integral(
                    circle, grid[i, j, k], 10)

    plot_magnetic_field(grid, B_field)


Emfield_plot_behavior_circle_wire()
