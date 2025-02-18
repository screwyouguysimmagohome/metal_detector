# Dette er en python fil, hvori jeg vil forsøge at modellere EM feldt omkring en circulær ledning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import grid3DFunc as grid3d
import scipy.integrate as integrate


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
    phi = np.arange(0, 360+step_size, step_size)*np.pi/180
    x_angles = np.cos(phi)*radius
    y_angles = np.sin(phi)*radius
    z_angles = np.zeros(x_angles.shape)*radius
    circle = np.column_stack((x_angles, y_angles, z_angles))

    return circle


circle_points = define_circle(1, 15)


def point_to_point_circle(circle_points):
    point_vecs = circle_points[1:, :3]-circle_points[:-1, :3]
    closing_vec = circle_points[0, :3]-circle_points[-1, :3]

    point_vecs = np.vstack([point_vecs, closing_vec])

    return point_vecs


point_to_point_vecs = point_to_point_circle(circle_points)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(122, projection='3d')

X = np.zeros(circle_points.shape[0])
Y = np.zeros(circle_points.shape[0])
Z = np.zeros(circle_points.shape[0])

U = circle_points[:, 0]
V = circle_points[:, 1]
W = circle_points[:, 2]

X1 = circle_points[0:-1, 0]
Y1 = circle_points[0:-1, 1]
Z1 = circle_points[0:-1, 2]

# Include last point
X1 = np.append(circle_points[:-1, 0], circle_points[-1, 0])
Y1 = np.append(circle_points[:-1, 1], circle_points[-1, 1])
Z1 = np.append(circle_points[:-1, 2], circle_points[-1, 2])


U1 = point_to_point_vecs[:, 0]
V1 = point_to_point_vecs[:, 1]
W1 = point_to_point_vecs[:, 2]

ax.quiver(X, Y, Z, U, V, W,
          color='r', label='circle points')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim([-1.2, 1.2])  # Extend slightly beyond the circle
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])
ax.set_title("wire circle 1")
ax.legend()


ax1.quiver(X1, Y1, Z1, U1, V1, W1,
           color='b', label='points to point vectors')
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_xlim([-1.2, 1.2])  # Extend slightly beyond the circle
ax1.set_ylim([-1.2, 1.2])
ax1.set_zlim([-1.2, 1.2])
ax1.set_title("wire circle 2")
ax1.legend()

plt.show()
