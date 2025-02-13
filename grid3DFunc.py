# lad os lave et grid med mange punkter og finde magnetfeltets retning og styrke i hver felt. for på den måde at få et indblik i magnetfeltets opførsel.
import numpy as np


def grid_3D(Xstart, Xstop, Xspaceing, Ystart, Ystop, Yspaceing, Zstart, Zstop, Zspaceing):

    x = np.arange(Xstart, Xstop+Xspaceing, Xspaceing)
    y = np.arange(Ystart, Ystop+Yspaceing, Yspaceing)
    z = np.arange(Zstart, Zstop+Zspaceing, Zspaceing)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    grid = np.stack((X, Y, Z), axis=-1)

    return grid
