"""
Step 4: Compare dn/dlnM from our JAX HMF vs CLASS Boltzmann.

Both use Tinker 2008 at Δ = 200 w.r.t. mean matter density
(which corresponds to M_200c with Δ_mean = 200/Ω_m(z)).

The CLASS Boltzmann comparison computes dn/dlnM from:
  1. P(k, z) from CLASS → σ(R, z) via TophatVar → R(M) → σ(M, z)
  2. Same Tinker 2008 formula applied to the Boltzmann σ(M, z)

Units: M in M_sun, dn/dlnM in 1/Mpc³.

Run:  python notebooks/step4_hmf.py
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

# ── 1. Our JAX HMF ──────────────────────────────────────────────────────────
print("=" * 60)
print("1. JAX HMF (cosmology.build + hmf.build_halo_grids)")
print("=" * 60)

from classy_szfast.cosmology import build
from classy_szfast.hmf import build_halo_grids, DELTA_C

t0 = time.time()
cg = build(cosmo, z_grid=jnp.array(z_test))
hg = build_halo_grids(cg, cosmo, delta_crit=200.0,
                       m_min=1e10, m_max=5e16, n_m=200)
t_emu = time.time() - t0

M_jax = np.exp(np.array(hg.lnM))          # M_sun
dndlnm_jax = np.array(hg.dndlnm)          # (n_z, n_m) 1/Mpc³
sigma_jax = np.array(hg.sigma_m)
rho_m0 = float(hg.rho_m0)

print(f"  M range: [{M_jax.min():.2e}, {M_jax.max():.2e}] M_sun")
print(f"  rho_m0 = {rho_m0:.4e} M_sun/Mpc³")
print(f"  elapsed = {t_emu:.2f}s")

for iz, z in enumerate(z_test):
    print(f"  z={z:.1f}: dn/dlnM(M=1e14 M_sun) = "
          f"{np.interp(np.log(1e14), np.log(M_jax), np.log(dndlnm_jax[iz] + 1e-300)):.2e} (log)")

# ── 2. CLASS Boltzmann ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CLASS Boltzmann → σ(R) → Tinker08 HMF")
print("=" * 60)

from classy_sz import Class as Class_sz
from mcfit import TophatVar

csz = Class_sz()
csz.set({**cosmo, 'output': 'mPk', 'P_k_max_1/Mpc': 50.0, 'z_max_pk': 5.5})
csz.compute()

# Physical constants (same as hmf.py)
G_SI = 6.67428e-11
Msun_kg = 1.98855e30
Mpc_m = 3.085677581282e22
c_SI = 2.99792458e8

# ρ_crit,0 and ρ̄_m
Omega_b = cosmo['omega_b'] / h**2
Omega_cdm = cosmo['omega_cdm'] / h**2
Omega_ncdm = 0.06 / (93.14 * h**2)
Omega_m = Omega_b + Omega_cdm + Omega_ncdm

H0_si = cosmo['H0'] * 1e3 / Mpc_m
rho_crit_0_boltz = 3.0 * H0_si**2 / (8.0 * np.pi * G_SI) * Mpc_m**3 / Msun_kg
rho_m0_boltz = Omega_m * rho_crit_0_boltz
print(f"  rho_m0 = {rho_m0_boltz:.4e} M_sun/Mpc³  (ratio to JAX: {rho_m0_boltz/rho_m0:.6f})")

# Same mass grid
M_grid = M_jax.copy()
R_lag = (3.0 * M_grid / (4.0 * np.pi * rho_m0_boltz))**(1.0/3.0)  # Mpc

print(f"  R_lag range: [{R_lag.min():.4f}, {R_lag.max():.1f}] Mpc")

# P(k, z) from CLASS → σ(R, z) via TophatVar
k_class = np.geomspace(1e-4, 50.0, 2000)  # 1/Mpc

sigma_boltz = np.zeros((len(z_test), len(M_grid)))
for iz, z in enumerate(z_test):
    pk_z = np.array([csz.pk_lin(float(kv), z) for kv in k_class])
    tv = TophatVar(k_class, lowring=True)
    R_tv, var_tv = tv(pk_z, extrap=True)
    # Interpolate σ at the Lagrangian radii
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(np.log(R_tv), np.log(var_tv))
    sigma_boltz[iz] = np.sqrt(np.exp(cs(np.log(R_lag))))

# Compute dn/dlnM from Boltzmann σ using same Tinker08
# dlnσ²/dlnR via finite differences
lnsig2_boltz = 2.0 * np.log(sigma_boltz)
dlnR = (1.0/3.0) * (np.log(M_grid[1]) - np.log(M_grid[0]))
dlnsig2_dlnR_boltz = np.gradient(lnsig2_boltz, dlnR, axis=1)

# H(z) and Omega_m(z) from CLASS
Hz_boltz = np.array([csz.Hubble(z) for z in z_test])  # H/c in 1/Mpc
Hz_si_boltz = Hz_boltz * c_SI / Mpc_m
rho_crit_z_boltz = 3.0 * Hz_si_boltz**2 / (8.0 * np.pi * G_SI) * Mpc_m**3 / Msun_kg
Omega_m_z_boltz = rho_m0_boltz * (1.0 + np.array(z_test))**3 / rho_crit_z_boltz
delta_mean_boltz = 200.0 / Omega_m_z_boltz

# Tinker08 parameters (same formula as hmf.py)
T08_DELTA_TAB = np.log10(np.array([200., 300., 400., 600., 800.,
                                     1200., 1600., 2400., 3200.]))
T08_A_TAB = np.array([0.186, 0.200, 0.212, 0.218, 0.248,
                        0.255, 0.260, 0.260, 0.260])
T08_a_TAB = np.array([1.47, 1.52, 1.56, 1.61, 1.87,
                        2.13, 2.30, 2.53, 2.66])
T08_b_TAB = np.array([2.57, 2.25, 2.05, 1.87, 1.59,
                        1.51, 1.46, 1.44, 1.41])
T08_c_TAB = np.array([1.19, 1.27, 1.34, 1.45, 1.58,
                        1.80, 1.97, 2.24, 2.44])

dndlnm_boltz = np.zeros_like(sigma_boltz)
for iz, z in enumerate(z_test):
    log_d = np.log10(delta_mean_boltz[iz])
    Ap = np.interp(log_d, T08_DELTA_TAB, T08_A_TAB) * (1+z)**(-0.14)
    a  = np.interp(log_d, T08_DELTA_TAB, T08_a_TAB) * (1+z)**(-0.06)
    b  = np.interp(log_d, T08_DELTA_TAB, T08_b_TAB) * (1+z)**(
            -10.0**(-((0.75/np.log10(delta_mean_boltz[iz]/75.0))**1.2)))
    c  = np.interp(log_d, T08_DELTA_TAB, T08_c_TAB)

    f_sigma = 0.5 * Ap * ((sigma_boltz[iz] / b)**(-a) + 1.0) * np.exp(-c / sigma_boltz[iz]**2)

    dndlnm_boltz[iz] = (rho_m0_boltz / M_grid
                         * f_sigma
                         * (1.0/3.0)
                         * np.abs(dlnsig2_dlnR_boltz[iz]))

# ── 3. Comparison table ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("σ(M, z) ratios: JAX / Boltzmann")
print("=" * 60)

M_check = [1e11, 1e12, 1e13, 1e14, 1e15]
print(f"\n{'M [Msun]':>12}", end="")
for z in z_test:
    print(f"  z={z:.1f}", end="")
print()
for mc in M_check:
    print(f"{mc:12.1e}", end="")
    for iz in range(len(z_test)):
        s_jax = np.interp(np.log(mc), np.log(M_jax), np.log(sigma_jax[iz]))
        s_bolt = np.interp(np.log(mc), np.log(M_grid), np.log(sigma_boltz[iz]))
        ratio = np.exp(s_jax) / np.exp(s_bolt)
        print(f"  {ratio:.4f}", end="")
    print()

print("\n" + "=" * 60)
print("dn/dlnM ratios: JAX / Boltzmann")
print("=" * 60)

print(f"\n{'M [Msun]':>12}", end="")
for z in z_test:
    print(f"  z={z:.1f}", end="")
print()
for mc in M_check:
    print(f"{mc:12.1e}", end="")
    for iz in range(len(z_test)):
        dn_jax = np.exp(np.interp(np.log(mc), np.log(M_jax), np.log(dndlnm_jax[iz] + 1e-300)))
        dn_bolt = np.exp(np.interp(np.log(mc), np.log(M_grid), np.log(dndlnm_boltz[iz] + 1e-300)))
        if dn_bolt > 0:
            ratio = dn_jax / dn_bolt
            print(f"  {ratio:.4f}", end="")
        else:
            print(f"     nan", end="")
    print()

# ── 4. Plot ──────────────────────────────────────────────────────────────────
label_size = 17
title_size = 18
legend_size = 11

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Top left: dn/dlnM vs M
ax_hmf = axes[0, 0]
# Top right: ratio dn/dlnM
ax_rat = axes[0, 1]
# Bottom left: σ(M)
ax_sig = axes[1, 0]
# Bottom right: ratio σ
ax_sigr = axes[1, 1]

for ax in axes.flat:
    ax.tick_params(axis='x', which='both', length=5, direction='in', pad=10)
    ax.tick_params(axis='y', which='both', length=5, direction='in', pad=5)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.setp(ax.get_yticklabels(), fontsize=label_size - 3)
    plt.setp(ax.get_xticklabels(), fontsize=label_size - 3)
    ax.grid(visible=True, which='both', alpha=0.2, linestyle='--')

colors = ['C0', 'C1', 'C2', 'C3']
for iz, z in enumerate(z_test):
    c = colors[iz]

    # dn/dlnM
    valid_j = dndlnm_jax[iz] > 0
    valid_b = dndlnm_boltz[iz] > 0
    ax_hmf.loglog(M_jax[valid_j], dndlnm_jax[iz, valid_j],
                  color=c, ls='-', lw=2, label=f'emu z={z}')
    ax_hmf.loglog(M_grid[valid_b], dndlnm_boltz[iz, valid_b],
                  color=c, ls='--', lw=1.5)

    # ratio (interpolated at common M)
    both = valid_j & valid_b
    if both.any():
        ratio = dndlnm_jax[iz, both] / dndlnm_boltz[iz, both]
        ax_rat.semilogx(M_jax[both], ratio, color=c, ls='-', lw=2,
                        label=f'z={z}')

    # σ(M)
    ax_sig.loglog(M_jax, sigma_jax[iz], color=c, ls='-', lw=2,
                  label=f'emu z={z}')
    ax_sig.loglog(M_grid, sigma_boltz[iz], color=c, ls='--', lw=1.5)

    # σ ratio
    ratio_s = sigma_jax[iz] / sigma_boltz[iz]
    ax_sigr.semilogx(M_jax, ratio_s, color=c, ls='-', lw=2,
                     label=f'z={z}')

ax_hmf.set_xlabel(r"$M\;[M_\odot]$", size=title_size)
ax_hmf.set_ylabel(r"$dn/d\ln M\;[{\rm Mpc}^{-3}]$", size=title_size)
ax_hmf.legend(fontsize=legend_size, handlelength=1.5)
ax_hmf.set_title("Solid=emulator, Dashed=Boltzmann", fontsize=13)

ax_rat.set_xlabel(r"$M\;[M_\odot]$", size=title_size)
ax_rat.set_ylabel("emu / Boltzmann", size=title_size)
ax_rat.axhline(1.0, color='k', ls=':', lw=0.5)
ax_rat.set_ylim(0.9, 1.1)
ax_rat.legend(fontsize=legend_size, handlelength=1.5)
ax_rat.set_title(r"$dn/d\ln M$ ratio", fontsize=13)

ax_sig.set_xlabel(r"$M\;[M_\odot]$", size=title_size)
ax_sig.set_ylabel(r"$\sigma(M, z)$", size=title_size)
ax_sig.legend(fontsize=legend_size, handlelength=1.5)

ax_sigr.set_xlabel(r"$M\;[M_\odot]$", size=title_size)
ax_sigr.set_ylabel("emu / Boltzmann", size=title_size)
ax_sigr.axhline(1.0, color='k', ls=':', lw=0.5)
ax_sigr.set_ylim(0.99, 1.01)
ax_sigr.legend(fontsize=legend_size, handlelength=1.5)
ax_sigr.set_title(r"$\sigma(M,z)$ ratio", fontsize=13)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__) or '.', 'step4_hmf.png')
plt.savefig(outpath, dpi=150)
print(f"\nPlot saved: {outpath}")

csz.struct_cleanup()
