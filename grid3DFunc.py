# lad os lave et grid med mange punkter og finde magnetfeltets retning og styrke i hver felt. for på den måde at få et indblik i magnetfeltets opførsel.
import numpy as np


def grid_3D(Xstart, Xstop, Xspaceing, Ystart, Ystop, Yspaceing, Zstart, Zstop, Zspaceing):
    """
    Generates a 3D grid of points in (x, y, z) space.

    Parameters:
        Xstart (float): The starting value for the x-axis.
        Xstop (float): The ending value for the x-axis.
        Xspacing (float): The spacing between points along the x-axis.
        Ystart (float): The starting value for the y-axis.
        Ystop (float): The ending value for the y-axis.
        Yspacing (float): The spacing between points along the y-axis.
        Zstart (float): The starting value for the z-axis.
        Zstop (float): The ending value for the z-axis.
        Zspacing (float): The spacing between points along the z-axis.

    Returns:
        numpy.ndarray: A 4D array of shape (N_x, N_y, N_z, 3), where each element 
                       represents the (x, y, z) coordinates of a grid point.
    """

    x = np.arange(Xstart, Xstop+Xspaceing, Xspaceing)
    y = np.arange(Ystart, Ystop+Xspaceing, Yspaceing)
    z = np.arange(Zstart, Zstop+Zspaceing, Zspaceing)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    grid = np.stack((X, Y, Z), axis=-1)

    return grid
