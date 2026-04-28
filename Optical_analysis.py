import numpy as np
import matplotlib.pyplot as plt
import csv
from pymatgen.io.vasp import Vasprun
from scipy.signal import savgol_filter

# =========================
# User options
# =========================
SMOOTH = True           # True = plot smoothed curves
WINDOW = 21             # must be odd
POLYORDER = 3           # usually 2 or 3

# Load vasprun.xml
vasprun = Vasprun("vasprun.xml", parse_dos=False)
dielectric_data = vasprun.dielectric  # [energies, eps_real, eps_imag]

energies = np.array(dielectric_data[0])   # shape (N,)
eps_real = np.array(dielectric_data[1])   # shape (N, 3)
eps_imag = np.array(dielectric_data[2])   # shape (N, 3)

# Extract dielectric tensor components
eps1_xx = eps_real[:, 0]
eps1_yy = eps_real[:, 1]
eps1_zz = eps_real[:, 2]

eps2_xx = eps_imag[:, 0]
eps2_yy = eps_imag[:, 1]
eps2_zz = eps_imag[:, 2]

# In-plane average and out-of-plane component
eps1_in = 0.5 * (eps1_xx + eps1_yy)
eps2_in = 0.5 * (eps2_xx + eps2_yy)
eps1_out = eps1_zz
eps2_out = eps2_zz

# Constants
hbar = 6.582119569e-16  # eV*s
c = 2.99792458e8        # m/s

# Angular frequency in rad/s
omega = energies / hbar   # E = ħω

def get_optics(eps1, eps2, omega):
    abs_eps = np.sqrt(eps1**2 + eps2**2)

    # protect against tiny negative numerical values inside sqrt
    n_term = (abs_eps + eps1) / 2.0
    k_term = (abs_eps - eps1) / 2.0

    n = np.sqrt(np.clip(n_term, 0, None))
    k = np.sqrt(np.clip(k_term, 0, None))

    # Absorption coefficient in 1/m
    alpha_m = 2.0 * omega * k / c

    # Convert to 1/cm
    alpha_cm = alpha_m * 1e-2

    # Reflectivity
    R = ((n - 1.0)**2 + k**2) / ((n + 1.0)**2 + k**2)

    return n, k, alpha_cm, R

def smooth_curve(y, window=21, polyorder=3):
    """
    Apply Savitzky-Golay smoothing if possible.
    Keeps original data if array is too short.
    """
    n = len(y)

    # window must be odd and <= n
    if n < 5:
        return y.copy()

    if window >= n:
        window = n - 1 if n % 2 == 0 else n

    if window % 2 == 0:
        window -= 1

    if window <= polyorder:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1

    if window > n:
        return y.copy()

    return savgol_filter(y, window_length=window, polyorder=polyorder)

# Calculate optical properties
n_in, k_in, alpha_in, R_in = get_optics(eps1_in, eps2_in, omega)
n_out, k_out, alpha_out, R_out = get_optics(eps1_out, eps2_out, omega)

# Save RAW data to CSV
rows = []
for i in range(len(energies)):
    rows.append({
        "energy_eV": energies[i],
        "eps1_in": eps1_in[i],
        "eps1_out": eps1_out[i],
        "eps2_in": eps2_in[i],
        "eps2_out": eps2_out[i],
        "n_in": n_in[i],
        "n_out": n_out[i],
        "k_in": k_in[i],
        "k_out": k_out[i],
        "alpha_in_cm^-1": alpha_in[i],
        "alpha_out_cm^-1": alpha_out[i],
        "R_in": R_in[i],
        "R_out": R_out[i],
    })

with open("output.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "energy_eV",
            "eps1_in", "eps1_out",
            "eps2_in", "eps2_out",
            "n_in", "n_out",
            "k_in", "k_out",
            "alpha_in_cm^-1", "alpha_out_cm^-1",
            "R_in", "R_out"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("CSV file created: output.csv")

# Smooth only for plotting
if SMOOTH:
    n_in_plot = smooth_curve(n_in, WINDOW, POLYORDER)
    n_out_plot = smooth_curve(n_out, WINDOW, POLYORDER)

    k_in_plot = smooth_curve(k_in, WINDOW, POLYORDER)
    k_out_plot = smooth_curve(k_out, WINDOW, POLYORDER)

    alpha_in_plot = smooth_curve(alpha_in, WINDOW, POLYORDER)
    alpha_out_plot = smooth_curve(alpha_out, WINDOW, POLYORDER)

    R_in_plot = smooth_curve(R_in, WINDOW, POLYORDER)
    R_out_plot = smooth_curve(R_out, WINDOW, POLYORDER)

    eps1_in_plot = smooth_curve(eps1_in, WINDOW, POLYORDER)
    eps1_out_plot = smooth_curve(eps1_out, WINDOW, POLYORDER)

    eps2_in_plot = smooth_curve(eps2_in, WINDOW, POLYORDER)
    eps2_out_plot = smooth_curve(eps2_out, WINDOW, POLYORDER)
else:
    n_in_plot, n_out_plot = n_in, n_out
    k_in_plot, k_out_plot = k_in, k_out
    alpha_in_plot, alpha_out_plot = alpha_in, alpha_out
    R_in_plot, R_out_plot = R_in, R_out
    eps1_in_plot, eps1_out_plot = eps1_in, eps1_out
    eps2_in_plot, eps2_out_plot = eps2_in, eps2_out

# -----------------------------
# Plot optical properties
# -----------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(energies, n_in_plot, label='n (in-plane)', color='blue')
plt.plot(energies, n_out_plot, label='n (out-of-plane)', color='green')
plt.title("Refractive Index")
plt.xlabel("Energy (eV)")
plt.ylabel("n")
plt.xlim(0, 7)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(energies, k_in_plot, label='k (in-plane)', color='blue')
plt.plot(energies, k_out_plot, label='k (out-of-plane)', color='green')
plt.title("Extinction Coefficient")
plt.xlabel("Energy (eV)")
plt.ylabel("k")
plt.xlim(0, 7)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(energies, alpha_in_plot, label='α (in-plane)', color='blue')
plt.plot(energies, alpha_out_plot, label='α (out-of-plane)', color='green')
plt.title("Absorption Coefficient")
plt.xlabel("Energy (eV)")
plt.ylabel("α (cm$^{-1}$)")
plt.xlim(0, 7)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(energies, R_in_plot, label='R (in-plane)', color='blue')
plt.plot(energies, R_out_plot, label='R (out-of-plane)', color='green')
plt.title("Optical Reflectivity")
plt.xlabel("Energy (eV)")
plt.ylabel("Reflectivity")
plt.xlim(0, 7)
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig("optical_properties_smoothed.png", dpi=300)
plt.show()

# -----------------------------
# Plot dielectric function
# -----------------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(energies, eps1_in_plot, label='ε₁ (in-plane)', color='blue')
plt.plot(energies, eps1_out_plot, label='ε₁ (out-of-plane)', color='green')
plt.title("Real Part of Dielectric Function")
plt.xlabel("Energy (eV)")
plt.ylabel("ε₁")
plt.xlim(0, 7)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(energies, eps2_in_plot, label='ε₂ (in-plane)', color='blue')
plt.plot(energies, eps2_out_plot, label='ε₂ (out-of-plane)', color='green')
plt.title("Imaginary Part of Dielectric Function")
plt.xlabel("Energy (eV)")
plt.ylabel("ε₂")
plt.xlim(0, 7)
plt.legend()

plt.tight_layout()
plt.savefig("dielectric_function_smoothed.png", dpi=300)
plt.show()