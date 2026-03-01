"""
Tutorial: compute C_ell^yy and its gradients with the JAX pipeline.

Three parts:
  1. Compute C_ell^yy (1-halo + 2-halo) with the Arnaud 2010 profile
  2. Take gradients w.r.t. cosmological parameters
  3. Take gradients w.r.t. pressure-profile parameters

Run:  python notebooks/tutorial_cl_yy.py
"""
import warnings; warnings.filterwarnings('ignore')
import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
import time

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classy_szfast.differentiable import (
    CosmoParams,
    ProfileParamsA10,
    cl_yy_from_params,
)


# =====================================================================
# 1. Forward pass — compute C_ell^yy
# =====================================================================
print("=" * 60)
print("1. Computing C_ell^yy")
print("=" * 60)

# Define cosmological parameters (Planck 2018-ish)
cosmo = CosmoParams(
    omega_b    = 0.02242,
    omega_cdm  = 0.11933,
    H0         = 67.66,
    tau_reio   = 0.0561,
    ln10_10_As = 3.047,
    n_s        = 0.9665,
)

# Arnaud 2010 profile (defaults — you can change these)
profile = ProfileParamsA10()

# Multipole range
ell = jnp.geomspace(2, 5000, 80)

# Compute
t0 = time.perf_counter()
cl_1h, cl_2h = cl_yy_from_params(
    ell, cosmo,
    profile_params=profile,
    profile='arnaud10',
    delta_crit=500.0,
    n_z=100,
    n_m=200,
)
dt = time.perf_counter() - t0

# Convert to D_ell × 10^12 for plotting
dl_fac = ell * (ell + 1) / (2 * jnp.pi) * 1e12
dl_1h = dl_fac * cl_1h
dl_2h = dl_fac * cl_2h

print(f"  Time:      {dt*1e3:.0f} ms")
print(f"  ell range: [{float(ell[0]):.0f}, {float(ell[-1]):.0f}]")
print(f"  D_ell 1h:  [{float(dl_1h.min()):.3e}, {float(dl_1h.max()):.3e}]")
print(f"  D_ell 2h:  [{float(dl_2h.min()):.3e}, {float(dl_2h.max()):.3e}]")


# =====================================================================
# 2. Gradients w.r.t. cosmological parameters
# =====================================================================
print("\n" + "=" * 60)
print("2. Gradients w.r.t. cosmological parameters")
print("=" * 60)

def loss_cosmo(c):
    """Scalar loss = sum of C_ell (1h + 2h)."""
    cl_1h, cl_2h = cl_yy_from_params(
        ell, c, profile_params=profile,
        profile='arnaud10', delta_crit=500.0,
    )
    return jnp.sum(cl_1h + cl_2h)

t0 = time.perf_counter()
grads_cosmo = jax.grad(loss_cosmo)(cosmo)
dt = time.perf_counter() - t0
print(f"  jax.grad time: {dt:.1f} s")

for name, val, g in zip(cosmo._fields, cosmo, grads_cosmo):
    print(f"  d(loss)/d({name:>14s}) = {float(g):+12.4e}   (value={float(val)})")


# =====================================================================
# 3. Gradients w.r.t. profile parameters
# =====================================================================
print("\n" + "=" * 60)
print("3. Gradients w.r.t. Arnaud 2010 profile parameters")
print("=" * 60)

def loss_profile(pp):
    """Scalar loss = sum of C_ell^1h (most sensitive to profile)."""
    cl_1h, _ = cl_yy_from_params(
        ell, cosmo, profile_params=pp,
        profile='arnaud10', delta_crit=500.0,
    )
    return jnp.sum(cl_1h)

t0 = time.perf_counter()
grads_profile = jax.grad(loss_profile)(profile)
dt = time.perf_counter() - t0
print(f"  jax.grad time: {dt:.1f} s")

for name, val, g in zip(profile._fields, profile, grads_profile):
    print(f"  d(loss)/d({name:>5s}) = {float(g):+12.4e}   (default={float(val)})")


# =====================================================================
# 4. Vary one parameter and plot the response
# =====================================================================
print("\n" + "=" * 60)
print("4. Parameter sweep: varying omega_cdm")
print("=" * 60)

omega_cdm_vals = jnp.linspace(0.10, 0.14, 5)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for omega_cdm in omega_cdm_vals:
    c = cosmo._replace(omega_cdm=float(omega_cdm))
    cl1, cl2 = cl_yy_from_params(
        ell, c, profile_params=profile,
        profile='arnaud10', delta_crit=500.0,
    )
    dl_tot = dl_fac * (cl1 + cl2)
    label = rf"$\omega_{{cdm}}={float(omega_cdm):.3f}$"
    axes[0].loglog(ell, dl_tot, lw=1.8, label=label)

axes[0].set_xlabel(r"$\ell$", fontsize=15)
axes[0].set_ylabel(r"$D_\ell^{yy} \times 10^{12}$", fontsize=15)
axes[0].set_title(r"Varying $\omega_{cdm}$", fontsize=14)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.2, ls='--')

# Also vary P0
P0_vals = jnp.array([5.0, 6.5, 8.13, 10.0, 12.0])
for P0 in P0_vals:
    pp = ProfileParamsA10(P0=float(P0))
    cl1, cl2 = cl_yy_from_params(
        ell, cosmo, profile_params=pp,
        profile='arnaud10', delta_crit=500.0,
    )
    dl_tot = dl_fac * (cl1 + cl2)
    label = rf"$P_0={float(P0):.1f}$"
    axes[1].loglog(ell, dl_tot, lw=1.8, label=label)

axes[1].set_xlabel(r"$\ell$", fontsize=15)
axes[1].set_ylabel(r"$D_\ell^{yy} \times 10^{12}$", fontsize=15)
axes[1].set_title(r"Varying $P_0$ (Arnaud 2010)", fontsize=14)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.2, ls='--')

for ax in axes:
    ax.tick_params(direction='in', which='both')

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__) or '.', 'tutorial_cl_yy.png')
plt.savefig(outpath, dpi=150)
print(f"\n  Plot saved: {outpath}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
