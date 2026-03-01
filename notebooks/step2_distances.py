"""
Step 2: Compare H(z), chi(z), d_A(z) from our emulator vs CLASS Boltzmann.

Units:
  H(z)  in km/s/Mpc
  chi(z) in Mpc
  d_A(z) in Mpc

Run:  python notebooks/step2_distances.py
"""
import warnings; warnings.filterwarnings('ignore')
import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np
import time

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Cosmology ────────────────────────────────────────────────────────────────
cosmo = {
    'omega_b': 0.02242, 'omega_cdm': 0.11933, 'H0': 67.66,
    'tau_reio': 0.0561, 'ln10^{10}A_s': 3.047, 'n_s': 0.9665,
}
h = cosmo['H0'] / 100.0
c_km_s = 299792.458

z_arr = np.linspace(0.01, 5.0, 200)

# ── 1. Our JAX emulator ─────────────────────────────────────────────────────
print("=" * 60)
print("1. JAX emulator (cosmology module)")
print("=" * 60)

from classy_szfast.cosmology import get_distances

Hz_emu, chi_emu, Da_emu = get_distances(cosmo, jnp.array(z_arr))
Hz_emu_kms = np.array(Hz_emu) * c_km_s   # convert to km/s/Mpc
chi_emu = np.array(chi_emu)
Da_emu = np.array(Da_emu)

print(f"  H(z=0.5)  = {np.interp(0.5, z_arr, Hz_emu_kms):.2f} km/s/Mpc")
print(f"  chi(z=0.5) = {np.interp(0.5, z_arr, chi_emu):.1f} Mpc")
print(f"  d_A(z=0.5) = {np.interp(0.5, z_arr, Da_emu):.1f} Mpc")

# ── 2. CLASS-SZ Boltzmann ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CLASS Boltzmann")
print("=" * 60)

from classy_sz import Class as Class_sz

csz = Class_sz()
csz.set({**cosmo, 'output': 'mPk', 'P_k_max_1/Mpc': 10.0, 'z_max_pk': 5.5})
csz.compute()

Hz_boltz = np.array([csz.Hubble(z) * c_km_s for z in z_arr])  # Hubble returns H/c in 1/Mpc
Da_boltz = np.array([csz.angular_distance(z) for z in z_arr])    # Mpc
chi_boltz = Da_boltz * (1.0 + z_arr)  # chi = d_A * (1+z)

print(f"  H(z=0.5)  = {np.interp(0.5, z_arr, Hz_boltz):.2f} km/s/Mpc")
print(f"  chi(z=0.5) = {np.interp(0.5, z_arr, chi_boltz):.1f} Mpc")
print(f"  d_A(z=0.5) = {np.interp(0.5, z_arr, Da_boltz):.1f} Mpc")

# ── 3. Ratios ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Ratios (emulator / Boltzmann)")
print("=" * 60)

r_Hz = Hz_emu_kms / Hz_boltz
r_chi = chi_emu / chi_boltz
r_Da = Da_emu / Da_boltz

print(f"  H(z):  mean={r_Hz.mean():.6f}  min={r_Hz.min():.6f}  max={r_Hz.max():.6f}")
print(f"  chi(z): mean={r_chi.mean():.6f}  min={r_chi.min():.6f}  max={r_chi.max():.6f}")
print(f"  d_A(z): mean={r_Da.mean():.6f}  min={r_Da.min():.6f}  max={r_Da.max():.6f}")

# ── 4. Plot ──────────────────────────────────────────────────────────────────
label_size = 17
title_size = 18
legend_size = 12

fig, axes = plt.subplots(3, 2, figsize=(12, 11))

quantities = [
    (Hz_emu_kms, Hz_boltz, r"$H(z)$ [km/s/Mpc]", r_Hz),
    (chi_emu, chi_boltz, r"$\chi(z)$ [Mpc]", r_chi),
    (Da_emu, Da_boltz, r"$d_A(z)$ [Mpc]", r_Da),
]

for irow, (q_emu, q_boltz, ylabel, ratio) in enumerate(quantities):
    ax_abs = axes[irow, 0]
    ax_rat = axes[irow, 1]

    for ax in [ax_abs, ax_rat]:
        ax.tick_params(axis='x', which='both', length=5, direction='in', pad=10)
        ax.tick_params(axis='y', which='both', length=5, direction='in', pad=5)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.setp(ax.get_yticklabels(), fontsize=label_size - 3)
        plt.setp(ax.get_xticklabels(), fontsize=label_size - 3)
        ax.grid(visible=True, which='both', alpha=0.2, linestyle='--')

    ax_abs.plot(z_arr, q_emu, 'C0-', lw=2, label='emulator')
    ax_abs.plot(z_arr, q_boltz, 'C1--', lw=2, label='Boltzmann')
    ax_abs.set_ylabel(ylabel, size=title_size - 2)
    ax_abs.legend(fontsize=legend_size)

    ax_rat.plot(z_arr, ratio, 'C0-', lw=2)
    ax_rat.axhline(1.0, color='k', ls=':', lw=0.5)
    ax_rat.set_ylabel("emu / boltz", size=title_size - 2)
    ax_rat.set_ylim(0.998, 1.002)

for ax in axes[-1, :]:
    ax.set_xlabel(r"$z$", size=title_size)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__) or '.', 'step2_distances.png')
plt.savefig(outpath, dpi=150)
print(f"\nPlot saved: {outpath}")

csz.struct_cleanup()
