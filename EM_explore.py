import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import grid3DFunc
from mpl_toolkits.mplot3d import Axes3D

mu0 = 4*np.pi*1e-7  # vaccum permeability
I = 10  # 10 ampere
wire_length = 0.1  # ledningslængde 10cm
wire_vector = [0, wire_length, 0]  # ud ad y-aksen


def biot_savart_cumsum(point, N_wire_segments):
    B = np.zeros(3)
    dl = wire_length/N_wire_segments

    for i in range(N_wire_segments):
        dy = (i+0.5)*dl
        segment_position = np.array([0, dy, 0])
        dl_vector = np.array([0, dl, 0])
        segment_to_point_vector = point-segment_position
        segment_to_point_vector_magnitude = np.linalg.norm(
            segment_to_point_vector)
        unit_vector = segment_to_point_vector/segment_to_point_vector_magnitude

        dB = (mu0/(4*np.pi))*I*np.cross(dl_vector, unit_vector) / \
            segment_to_point_vector_magnitude**2
        B += dB
    return B


grid = grid3DFunc.grid_3D(
    -0.05, 0.05, 0.002, -0.01, 0.11, 0.05, -0.05, 0.05, 0.02)

B_field = np.zeros(grid.shape[:3]+(3,))

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        for k in range(grid.shape[2]):
            B_field[i, j, k] = biot_savart_cumsum(grid[i, j, k], 100)


def plot_magnetic_field(grid, B_field):
    """
    Plotter magnetfeltet præcist som beregnet uden ændringer, hvor vektorernes længde afspejler magnetfeltets styrke.

    grid: 4D-array med (Nx, Ny, Nz, 3) koordinater for hvert punkt.
    B_field: 4D-array med (Nx, Ny, Nz, 3) magnetfelt-vektorer.
    """
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

    # Brug quiver til at tegne vektorer, hvor længden svarer til magnetfeltets styrke
    ax.quiver(X, Y, Z, U, V, W, length=np.max(
        B_magnitude), normalize=False, color='b')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Magnetfeltets retning og styrke i 3D")

    plt.show()


# Kald funktionen for at plotte B-feltet
# plot_magnetic_field(grid, B_field)
print(np.linalg.norm(B_field[0, :], axis=-1))

print(np.size(grid))
