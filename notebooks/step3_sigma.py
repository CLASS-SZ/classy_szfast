"""
Step 3: Compare sigma(R, z) from our emulator vs CLASS-SZ Boltzmann.

Also compare sigma8 (sigma at R=8 Mpc/h).

Units: R in Mpc, sigma dimensionless.

Run:  python notebooks/step3_sigma.py
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

z_test = [0.0, 0.5, 1.0, 2.0]
R_check = [1.0, 5.0, 8.0/h, 10.0, 20.0, 50.0]  # Mpc (8/h is for sigma8)

# ── 1. Our JAX emulator ─────────────────────────────────────────────────────
print("=" * 60)
print("1. JAX emulator (cosmology.get_sigma)")
print("=" * 60)

from classy_szfast.cosmology import get_sigma

t0 = time.time()
R_emu, sigma_emu, dsig2_emu = get_sigma(cosmo, jnp.array(z_test))
t_emu = time.time() - t0

R_np = np.array(R_emu)
sigma_np = np.array(sigma_emu)  # (n_z, n_R)

print(f"  R range: [{R_np.min():.4f}, {R_np.max():.1f}] Mpc")
print(f"  n_R = {len(R_np)},  elapsed = {t_emu:.2f}s")

# sigma8 = sigma(R = 8 Mpc/h)
R8 = 8.0 / h  # Mpc
for iz, z in enumerate(z_test):
    s8 = np.interp(np.log(R8), np.log(R_np), np.log(sigma_np[iz]))
    print(f"  sigma8(z={z:.1f}) = {np.exp(s8):.4f}")

# ── 2. CLASS-SZ Boltzmann ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CLASS Boltzmann")
print("=" * 60)

from classy_sz import Class as Class_sz

csz = Class_sz()
csz.set({**cosmo, 'output': 'mPk', 'P_k_max_1/Mpc': 50.0, 'z_max_pk': 5.5})
csz.compute()

# sigma(R, z) from CLASS: need to compute manually
# CLASS pk_lin(k, z) gives P(k) in Mpc^3, k in 1/Mpc
# sigma^2(R) = (1/2pi^2) int dk k^2 P(k) |W(kR)|^2
# where W(x) = 3(sin(x) - x cos(x))/x^3

# Use TophatVar with CLASS P(k)
from mcfit import TophatVar

k_class = np.geomspace(1e-4, 50.0, 2000)  # 1/Mpc
sigma_boltz = np.zeros((len(z_test), len(R_check)))
for iz, z in enumerate(z_test):
    pk_z = np.array([csz.pk_lin(float(kv), z) for kv in k_class])
    tv = TophatVar(k_class, lowring=True)
    R_tv, var_tv = tv(pk_z, extrap=True)
    # Interpolate at R_check
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(np.log(R_tv), np.log(var_tv))
    for ir, rv in enumerate(R_check):
        sigma_boltz[iz, ir] = np.sqrt(np.exp(cs(np.log(rv))))

print(f"  sigma8(z=0) = {sigma_boltz[0, 2]:.4f}")
for iz, z in enumerate(z_test):
    print(f"  sigma8(z={z:.1f}) = {sigma_boltz[iz, 2]:.4f}")

# ── 3. Comparison table ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("sigma(R, z) ratios: emulator / Boltzmann")
print("=" * 60)

sigma_emu_at_check = np.zeros((len(z_test), len(R_check)))
for iz in range(len(z_test)):
    for ir, rv in enumerate(R_check):
        sigma_emu_at_check[iz, ir] = np.exp(
            np.interp(np.log(rv), np.log(R_np), np.log(sigma_np[iz])))

print(f"\n{'R [Mpc]':>10}", end="")
for z in z_test:
    print(f"  z={z:.1f}", end="")
print()
for ir, rv in enumerate(R_check):
    label = f"{rv:.2f}" if rv < 100 else f"{rv:.0f}"
    if abs(rv - 8.0/h) < 0.01:
        label = f"8/h={rv:.2f}"
    print(f"{label:>10}", end="")
    for iz in range(len(z_test)):
        ratio = sigma_emu_at_check[iz, ir] / sigma_boltz[iz, ir]
        print(f"  {ratio:.4f}", end="")
    print()

# ── 4. Plot ──────────────────────────────────────────────────────────────────
label_size = 17
title_size = 18
legend_size = 12

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax_sig = axes[0]
ax_rat = axes[1]

for ax in axes:
    ax.tick_params(axis='x', which='both', length=5, direction='in', pad=10)
    ax.tick_params(axis='y', which='both', length=5, direction='in', pad=5)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.setp(ax.get_yticklabels(), fontsize=label_size)
    plt.setp(ax.get_xticklabels(), fontsize=label_size)
    ax.grid(visible=True, which='both', alpha=0.2, linestyle='--')

colors = ['C0', 'C1', 'C2', 'C3']
for iz, z in enumerate(z_test):
    c = colors[iz]
    # Emulator
    ax_sig.loglog(R_np, sigma_np[iz], color=c, ls='-', lw=2,
                  label=f'emu z={z}')
    # Boltzmann points
    ax_sig.scatter(R_check, sigma_boltz[iz], color=c, marker='x', s=80, zorder=5)

    # Ratio at R_check
    ratios = sigma_emu_at_check[iz] / sigma_boltz[iz]
    ax_rat.semilogx(R_check, ratios, color=c, ls='-', marker='o', lw=2,
                    label=f'z={z}')

ax_sig.set_xlabel(r"$R\;[\mathrm{Mpc}]$", size=title_size)
ax_sig.set_ylabel(r"$\sigma(R, z)$", size=title_size)
ax_sig.legend(fontsize=legend_size, handlelength=1.5)
ax_sig.set_title("Lines=emulator, Crosses=Boltzmann", fontsize=13)

ax_rat.set_xlabel(r"$R\;[\mathrm{Mpc}]$", size=title_size)
ax_rat.set_ylabel("emulator / Boltzmann", size=title_size)
ax_rat.axhline(1.0, color='k', ls=':', lw=0.5)
ax_rat.set_ylim(0.99, 1.01)
ax_rat.legend(fontsize=legend_size, handlelength=1.5)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__) or '.', 'step3_sigma.png')
plt.savefig(outpath, dpi=150)
print(f"\nPlot saved: {outpath}")

csz.struct_cleanup()
