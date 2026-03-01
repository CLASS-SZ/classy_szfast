"""
Step 6 — End-to-end differentiable C_ell^yy pipeline.

Verifies that jax.grad works through the full pipeline:
  emulators → sigma(R) → HMF → C_ell integration

Three checks:
  1. Forward pass matches the existing (non-differentiable) pipeline
  2. Gradient w.r.t. omega_b agrees with finite differences (<1%)
  3. Gradients w.r.t. profile params are finite and non-zero
"""

import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from classy_szfast.differentiable import (
    CosmoParams,
    ProfileParamsA10,
    cl_yy_from_params,
)
from classy_szfast.cosmology import build as build_cosmo_grids
from classy_szfast.hmf import build_halo_grids
from classy_szfast.power_spectrum import cl_yy_1h_2h


# ── Reference cosmology ──────────────────────────────────────────────

cosmo = CosmoParams(
    omega_b=0.02242,
    omega_cdm=0.11933,
    H0=67.66,
    tau_reio=0.0561,
    ln10_10_As=3.047,
    n_s=0.9665,
)

ell = jnp.geomspace(2, 3000, 50)

# =====================================================================
# 1. Forward-pass consistency
# =====================================================================
print("=" * 60)
print("1. Forward-pass consistency check")
print("=" * 60)

# New differentiable path
t0 = time.perf_counter()
cl_1h_new, cl_2h_new = cl_yy_from_params(ell, cosmo, profile='arnaud10',
                                          delta_crit=500.0)
dt_new = time.perf_counter() - t0
print(f"   cl_yy_from_params:  {dt_new*1e3:.1f} ms")

# Old path (same underlying functions, but called manually)
params = {
    'omega_b': 0.02242,
    'omega_cdm': 0.11933,
    'H0': 67.66,
    'tau_reio': 0.0561,
    'ln10^{10}A_s': 3.047,
    'n_s': 0.9665,
    'm_ncdm': 0.06,
}
t0 = time.perf_counter()
cg = build_cosmo_grids(params)
hg = build_halo_grids(cg, params, delta_crit=500.0)
cl_1h_old, cl_2h_old = cl_yy_1h_2h(ell, cg, hg, params, profile='arnaud10')
dt_old = time.perf_counter() - t0
print(f"   manual pipeline:    {dt_old*1e3:.1f} ms")

rtol_1h = jnp.max(jnp.abs(cl_1h_new / cl_1h_old - 1.0))
rtol_2h = jnp.max(jnp.abs(cl_2h_new / cl_2h_old - 1.0))
print(f"   max |cl_1h_new/cl_1h_old - 1| = {rtol_1h:.2e}")
print(f"   max |cl_2h_new/cl_2h_old - 1| = {rtol_2h:.2e}")
assert rtol_1h < 1e-6, f"1-halo mismatch: {rtol_1h}"
assert rtol_2h < 1e-6, f"2-halo mismatch: {rtol_2h}"
print("   PASSED")

# =====================================================================
# 2. Gradient w.r.t. omega_b (finite-difference check)
# =====================================================================
print()
print("=" * 60)
print("2. Gradient w.r.t. omega_b (AD vs finite differences)")
print("=" * 60)


def loss_omega_b(omega_b):
    c = cosmo._replace(omega_b=omega_b)
    cl_1h, cl_2h = cl_yy_from_params(ell, c, profile='arnaud10',
                                      delta_crit=500.0)
    return jnp.sum(cl_1h + cl_2h)


omega_b_0 = 0.02242
eps = 1e-5

# AD gradient
t0 = time.perf_counter()
grad_ad = jax.grad(loss_omega_b)(omega_b_0)
dt_ad = time.perf_counter() - t0
print(f"   jax.grad:           {grad_ad:.6e}  ({dt_ad:.1f} s)")

# Finite-difference gradient
t0 = time.perf_counter()
f_plus = loss_omega_b(omega_b_0 + eps)
f_minus = loss_omega_b(omega_b_0 - eps)
grad_fd = (f_plus - f_minus) / (2.0 * eps)
dt_fd = time.perf_counter() - t0
print(f"   finite diff:        {grad_fd:.6e}  ({dt_fd:.1f} s)")

rel_err = jnp.abs(grad_ad / grad_fd - 1.0)
print(f"   relative error:     {rel_err:.4e}")
assert rel_err < 0.01, f"AD/FD mismatch: {rel_err:.4e}"
print("   PASSED (<1%)")

# =====================================================================
# 3. Gradients w.r.t. profile params
# =====================================================================
print()
print("=" * 60)
print("3. Gradients w.r.t. Arnaud 2010 profile parameters")
print("=" * 60)


def loss_profile(pp):
    cl_1h, cl_2h = cl_yy_from_params(ell, cosmo, profile_params=pp,
                                      profile='arnaud10', delta_crit=500.0)
    return jnp.sum(cl_1h)


pp0 = ProfileParamsA10()

t0 = time.perf_counter()
grads = jax.grad(loss_profile)(pp0)
dt = time.perf_counter() - t0
print(f"   jax.grad time: {dt:.1f} s")

for name, g in grads._asdict().items():
    status = "ok" if jnp.isfinite(g) and g != 0 else "FAIL"
    print(f"   d(sum cl_1h)/d({name:>5s}) = {g:+.6e}  [{status}]")
    assert jnp.isfinite(g) and g != 0, f"Gradient for {name} is {g}"

print("   PASSED (all finite and non-zero)")

# =====================================================================
# 4. Jacobian: all cosmo params at once
# =====================================================================
print()
print("=" * 60)
print("4. Jacobian w.r.t. all cosmological parameters")
print("=" * 60)


def loss_cosmo(c):
    cl_1h, cl_2h = cl_yy_from_params(ell, c, profile='arnaud10',
                                      delta_crit=500.0)
    return jnp.sum(cl_1h + cl_2h)


t0 = time.perf_counter()
grads_cosmo = jax.grad(loss_cosmo)(cosmo)
dt = time.perf_counter() - t0
print(f"   jax.grad time: {dt:.1f} s")

for name, g in grads_cosmo._asdict().items():
    status = "ok" if jnp.isfinite(g) else "FAIL"
    print(f"   d(loss)/d({name:>14s}) = {g:+.6e}  [{status}]")
    assert jnp.isfinite(g), f"Gradient for {name} is {g}"

print("   PASSED (all finite)")

print()
print("=" * 60)
print("All checks passed.")
print("=" * 60)
