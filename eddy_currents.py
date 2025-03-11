import numpy as np

# Constants
B = 1e-5  # Tesla (Magnetic field strength)
mu0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
sigma_gold = 4.11e7  # Conductivity of gold (S/m)

# Material properties
material_radius = 0.1  # Meters (10 cm)
frekvens = 100000  # Hz (100 kHz)
omega = 2 * np.pi * frekvens  # Angular frequency

# Cross-sectional area of eddy currents (assuming circular loop)
A = np.pi * material_radius**2

# Compute Induced EMF using Faraday's Law
epsilon_max = omega * B * A
print(f"Induced EMF (ε_max): {epsilon_max:} V")

# Compute Skin Depth
denominator = mu0 * sigma_gold * omega
nominator = 2
val = nominator/denominator
skin_depth_material = np.sqrt(val)
print(f"Skin Depth (δ): {skin_depth_material:} m")

# Compute Eddy Current Resistance
length_current_path = 2 * np.pi * material_radius  # Circumference of loop
R_gold = length_current_path / \
    (sigma_gold * material_radius * skin_depth_material)
print(f"Resistance of Gold Loop (R): {R_gold:} Ω")

# Compute Eddy Current Magnitude
I_eddy = epsilon_max / R_gold
print(f"Eddy Current (I_eddy): {I_eddy:} A")
