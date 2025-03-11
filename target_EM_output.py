import cupy as cp
import numpy as np  # Keep this for Matplotlib conversions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mu0 = 4 * cp.pi * 1e-7  # Vacuum permeability


def grid_3D(Xstart, Xstop, Xspaceing, Ystart, Ystop, Yspaceing, Zstart, Zstop, Zspaceing):
    x = cp.arange(Xstart, Xstop + Xspaceing, Xspaceing)
    y = cp.arange(Ystart, Ystop + Yspaceing, Yspaceing)
    z = cp.arange(Zstart, Zstop + Zspaceing, Zspaceing)
    X, Y, Z = cp.meshgrid(x, y, z, indexing="xy")
    return cp.stack((X, Y, Z), axis=-1)


def define_circle(radius, definition):
    step_size = 360 / definition
    phi = cp.arange(step_size, 360 + step_size, step_size) * cp.pi / 180
    x = cp.cos(phi) * radius
    y = cp.sin(phi) * radius
    z = cp.zeros_like(x)
    return cp.column_stack((x, y, z))


def biot_savart_integral(wire_points, field_points, I):
    wire_points = cp.asarray(wire_points)
    field_points = cp.asarray(field_points)

    dl = wire_points[1:] - wire_points[:-1]  # Shape: (N_segments, 3)
    # Shape: (N_segments, 3)
    r_seg = 0.5 * (wire_points[1:] + wire_points[:-1])

    # Correctly reshape field_points to match segment positions
    field_points = field_points[..., None, :]  # Shape: (Nx, Ny, Nz, 1, 3)
    r_seg = r_seg[None, None, None, :, :]  # Shape: (1, 1, 1, N_segments, 3)

    r_vec = field_points - r_seg  # Shape: (Nx, Ny, Nz, N_segments, 3)
    r_mag = cp.linalg.norm(r_vec, axis=-1, keepdims=True)
    r_mag = cp.maximum(r_mag, 1e-4)  # Avoid division by zero

    # Shape: (Nx, Ny, Nz, N_segments, 3)
    cross_term = cp.cross(dl[None, None, None, :, :], r_vec, axis=-1)
    # Sum over segments → Shape: (Nx, Ny, Nz, 3)
    B_total = cp.sum(cross_term / r_mag**3, axis=3)

    return (mu0 * I / (4 * cp.pi)) * B_total, dl


def plot_magnetic_field(grid, B_field, circle_points, dl):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    X, Y, Z = cp.asnumpy(grid[..., 0]), cp.asnumpy(
        grid[..., 1]), cp.asnumpy(grid[..., 2])
    U, V, W = cp.asnumpy(B_field[..., 0]), cp.asnumpy(
        B_field[..., 1]), cp.asnumpy(B_field[..., 2])

    ax.quiver(X, Y, Z, U*1e1, V*1e1, W*1e1,
              length=1.0, normalize=False, color='r')
    ax.set_title("Magnetic Field in 3D")
    plt.show()


def Emfield_plot_behavior_circle_wire():
    grid = grid_3D(-0.04, 0.04, 0.005, -0.04, 0.04, 0.005, -0.08, 0.08, 0.005)

    B_field, dl = biot_savart_integral(define_circle(0.01, 720), grid, 10)
    plot_magnetic_field(grid, B_field, define_circle(0.01, 720), dl)


Emfield_plot_behavior_circle_wire()
