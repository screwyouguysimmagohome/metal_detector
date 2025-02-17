# Dette er en python fil, hvori jeg vil forsøge at modellere EM feldt omkring en circulær ledning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp

I = 10  # current


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
# circle = define_circle(0.05, 360)
