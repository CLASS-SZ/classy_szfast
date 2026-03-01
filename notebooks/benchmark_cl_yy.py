"""
Benchmark: JAX differentiable pipeline vs CLASS-SZ + timing profile.

1. Run CLASS-SZ (GNFW / A10, T08M500c) as ground truth
2. Run our JAX pipeline (old API + new differentiable API)
3. Profile each stage: emulators → sigma → HMF → C_ell → grad
4. Plot comparison + ratios

Run:  python notebooks/benchmark_cl_yy.py
"""
import warnings; warnings.filterwarnings('ignore')
import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
import time

import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Cosmology (same as step5) ────────────────────────────────────────

cosmo_dict = {
    'omega_b': 0.02, 'omega_cdm': 0.12, 'H0': 80.,
    'tau_reio': 0.0561, 'ln10^{10}A_s': 3.047, 'n_s': 0.9665,
}
h = cosmo_dict['H0'] / 100.0

z_min, z_max = 0.005, 2.0
M_min_h, M_max_h = 5e10, 3.5e15   # M_sun/h
m_min = M_min_h / h
m_max = M_max_h / h

# =====================================================================
# 1. CLASS-SZ reference
# =====================================================================
print("=" * 65)
print("1. CLASS-SZ (GNFW / A10, T08M500c, fast mode)")
print("=" * 65)

from classy_sz import Class as Class_sz

t0 = time.perf_counter()
csz = Class_sz()
csz.set(cosmo_dict)
csz.set({
    'output': 'tSZ_tSZ_1h,tSZ_tSZ_2h',
    'skip_input': 0, 'cosmo_model': 0,
    'ell_min': 2, 'ell_max': 8000,
    'dell': 0., 'dlogell': 0.2,
    'z_min': z_min, 'z_max': z_max,
    'M_min': M_min_h, 'M_max': M_max_h,
    'mass_function': 'T08M500c',
    'pressure_profile': 'GNFW',
    'P0GNFW': 8.130, 'c500': 1.156,
    'gammaGNFW': 0.3292, 'alphaGNFW': 1.0620, 'betaGNFW': 5.4807,
    'B': 1.0,
    'no_spline_in_tinker': 1,
    'HMF_prescription_NCDM': 1,
    'use_resnet_pkl': 0,
    'x_outSZ': 4.,
    'use_fft_for_profiles_transform': 1,
    'N_samp_fftw': 1024,
    'x_min_gas_pressure_fftw': 1e-4,
    'x_max_gas_pressure_fftw': 1e6,
    'n_m_pressure_profile': 100,
    'n_z_pressure_profile': 100,
    'ndim_redshifts': 150,
    'redshift_epsabs': 1e-40, 'redshift_epsrel': 1e-4,
    'mass_epsabs': 1e-40, 'mass_epsrel': 1e-4,
})
csz.initialize_classy_szfast()
t_csz = time.perf_counter() - t0

ell_csz = np.asarray(csz.cl_sz()['ell'])
dl_1h_csz = np.asarray(csz.cl_sz()['1h'])
dl_2h_csz = np.asarray(csz.cl_sz()['2h'])
csz.struct_cleanup()

print(f"  Total:     {t_csz*1e3:.0f} ms")
print(f"  n_ell:     {len(ell_csz)}")


# =====================================================================
# 2. JAX pipeline — stage-by-stage timing
# =====================================================================
print("\n" + "=" * 65)
print("2. JAX pipeline — stage-by-stage profiling")
print("=" * 65)

from classy_szfast.cosmology import build as build_cosmo_grids
from classy_szfast.hmf import build_halo_grids
from classy_szfast.power_spectrum import cl_yy_1h_2h

ell_jax = jnp.array(ell_csz)
z_grid = jnp.geomspace(z_min, z_max, 150)

# --- Warm-up run (JIT compilation, emulator loading) ---
print("\n  Warm-up run (includes JIT + emulator load)...")
t0 = time.perf_counter()
cg = build_cosmo_grids(cosmo_dict, z_grid=z_grid)
hg = build_halo_grids(cg, cosmo_dict, delta_crit=500.0,
                       m_min=m_min, m_max=m_max, n_m=300)
cl_1h, cl_2h = cl_yy_1h_2h(ell_jax, cg, hg, cosmo_dict, profile='arnaud10')
# Force computation
cl_1h.block_until_ready(); cl_2h.block_until_ready()
t_warmup = time.perf_counter() - t0
print(f"  Warm-up:   {t_warmup*1e3:.0f} ms")

# --- Timed run (hot caches) ---
print("\n  Timed run (hot caches)...")

t0 = time.perf_counter()
cg = build_cosmo_grids(cosmo_dict, z_grid=z_grid)
cg.pk.block_until_ready()
t_emu = time.perf_counter() - t0

t0 = time.perf_counter()
hg = build_halo_grids(cg, cosmo_dict, delta_crit=500.0,
                       m_min=m_min, m_max=m_max, n_m=300)
hg.dndlnm.block_until_ready()
t_hmf = time.perf_counter() - t0

t0 = time.perf_counter()
cl_1h, cl_2h = cl_yy_1h_2h(ell_jax, cg, hg, cosmo_dict, profile='arnaud10')
cl_1h.block_until_ready(); cl_2h.block_until_ready()
t_cl = time.perf_counter() - t0

t_total = t_emu + t_hmf + t_cl

print(f"  Emulators + sigma:  {t_emu*1e3:7.1f} ms")
print(f"  HMF + bias:         {t_hmf*1e3:7.1f} ms")
print(f"  C_ell integration:  {t_cl*1e3:7.1f} ms")
print(f"  ─────────────────────────────")
print(f"  Total (hot):        {t_total*1e3:7.1f} ms")
print(f"  CLASS-SZ:           {t_csz*1e3:7.0f} ms")
print(f"  Speedup:            {t_csz/t_total:7.1f}x")


# =====================================================================
# 3. Differentiable API — forward + gradient timing
# =====================================================================
print("\n" + "=" * 65)
print("3. Differentiable API — forward + gradient timing")
print("=" * 65)

from classy_szfast.differentiable import (
    CosmoParams, ProfileParamsA10, cl_yy_from_params,
)

cosmo_nt = CosmoParams(
    omega_b=0.02, omega_cdm=0.12, H0=80.,
    tau_reio=0.0561, ln10_10_As=3.047, n_s=0.9665,
)
profile_nt = ProfileParamsA10()

# Forward pass
t0 = time.perf_counter()
cl_1h_d, cl_2h_d = cl_yy_from_params(
    ell_jax, cosmo_nt, profile_params=profile_nt,
    profile='arnaud10', delta_crit=500.0,
    n_z=150, m_min=m_min, m_max=m_max, n_m=300,
)
cl_1h_d.block_until_ready(); cl_2h_d.block_until_ready()
t_fwd = time.perf_counter() - t0
print(f"  Forward (cl_yy_from_params): {t_fwd*1e3:.0f} ms")

# Check differentiable API matches manual pipeline
dl_fac = ell_jax * (ell_jax + 1) / (2 * jnp.pi) * 1e12
dl_1h_jax = np.array(dl_fac * cl_1h)
dl_2h_jax = np.array(dl_fac * cl_2h)
dl_1h_diff = np.array(dl_fac * cl_1h_d)
dl_2h_diff = np.array(dl_fac * cl_2h_d)

r_check = np.max(np.abs(dl_1h_diff / dl_1h_jax - 1))
print(f"  Differentiable vs manual:    max|ratio-1| = {r_check:.2e}")

# Gradient timing
def loss(c):
    cl1, cl2 = cl_yy_from_params(
        ell_jax, c, profile_params=profile_nt,
        profile='arnaud10', delta_crit=500.0,
        n_z=150, m_min=m_min, m_max=m_max, n_m=300,
    )
    return jnp.sum(cl1 + cl2)

# Warm up grad
print("\n  Gradient warm-up ...")
t0 = time.perf_counter()
g_warmup = jax.grad(loss)(cosmo_nt)
_ = g_warmup.omega_b.block_until_ready()
t_grad_warmup = time.perf_counter() - t0
print(f"  Grad warm-up:   {t_grad_warmup:.1f} s")

# Timed grad
t0 = time.perf_counter()
grads = jax.grad(loss)(cosmo_nt)
_ = grads.omega_b.block_until_ready()
t_grad = time.perf_counter() - t0
print(f"  Grad (hot):     {t_grad:.1f} s")

print(f"\n  Gradient values:")
for name, g in grads._asdict().items():
    print(f"    d(loss)/d({name:>14s}) = {float(g):+.4e}")


# =====================================================================
# 4. Comparison: JAX vs CLASS-SZ
# =====================================================================
print("\n" + "=" * 65)
print("4. Accuracy: JAX / CLASS-SZ ratios")
print("=" * 65)

r1h = dl_1h_jax / dl_1h_csz
r2h = dl_2h_jax / dl_2h_csz

print(f"  1h: mean={r1h.mean():.4f}  min={r1h.min():.4f}  max={r1h.max():.4f}")
print(f"  2h: mean={r2h.mean():.4f}  min={r2h.min():.4f}  max={r2h.max():.4f}")

print(f"\n  {'ell':>6}  {'JAX 1h':>11}  {'CSZ 1h':>11}  {'ratio':>7}"
      f"  {'JAX 2h':>11}  {'CSZ 2h':>11}  {'ratio':>7}")
for i in range(0, len(ell_csz), max(1, len(ell_csz) // 12)):
    e = float(ell_csz[i])
    print(f"  {e:6.0f}  {dl_1h_jax[i]:11.4e}  {dl_1h_csz[i]:11.4e}  {r1h[i]:7.4f}"
          f"  {dl_2h_jax[i]:11.4e}  {dl_2h_csz[i]:11.4e}  {r2h[i]:7.4f}")


# =====================================================================
# 5. Plot
# =====================================================================
print("\n" + "=" * 65)
print("5. Plotting")
print("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                          gridspec_kw={'height_ratios': [2, 1]})

# ── Top-left: D_ell spectra ──────────────────────────────────────────
ax = axes[0, 0]
ax.loglog(ell_csz, dl_1h_csz, 'C0--', lw=1.5, label='CLASS-SZ 1h', alpha=0.8)
ax.loglog(ell_csz, dl_2h_csz, 'C1--', lw=1.5, label='CLASS-SZ 2h', alpha=0.8)
ax.loglog(ell_csz, dl_1h_csz + dl_2h_csz, 'k--', lw=1.5, label='CLASS-SZ total', alpha=0.6)
ax.loglog(ell_csz, dl_1h_jax, 'C0-', lw=2, label='JAX 1h')
ax.loglog(ell_csz, dl_2h_jax, 'C1-', lw=2, label='JAX 2h')
ax.loglog(ell_csz, dl_1h_jax + dl_2h_jax, 'k-', lw=2, label='JAX total')
ax.set_ylabel(r"$D_\ell^{yy} \times 10^{12}$", fontsize=16)
ax.set_title("Arnaud 2010 (GNFW), Tinker 08 M500c", fontsize=14)
ax.legend(fontsize=9, ncol=2)

# ── Bottom-left: ratio ───────────────────────────────────────────────
ax = axes[1, 0]
ax.semilogx(ell_csz, r1h, 'C0-', lw=2, label='1-halo')
ax.semilogx(ell_csz, r2h, 'C1-', lw=2, label='2-halo')
ax.axhline(1.0, color='k', ls=':', lw=0.5)
ax.set_xlabel(r"$\ell$", fontsize=16)
ax.set_ylabel("JAX / CLASS-SZ", fontsize=14)
ax.legend(fontsize=10)

# ── Top-right: timing bar chart ──────────────────────────────────────
ax = axes[0, 1]
stages = ['Emulators\n+ sigma', 'HMF\n+ bias', r'$C_\ell$'+'\nintegration', 'Total\n(JAX)', 'CLASS-SZ']
times_ms = [t_emu*1e3, t_hmf*1e3, t_cl*1e3, t_total*1e3, t_csz*1e3]
colors = ['C0', 'C1', 'C2', 'C4', 'C3']
bars = ax.bar(stages, times_ms, color=colors, edgecolor='k', lw=0.5)
for bar, t in zip(bars, times_ms):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times_ms)*0.01,
            f'{t:.0f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel("Time (ms)", fontsize=14)
ax.set_title("Forward pass timing (hot caches)", fontsize=14)

# ── Bottom-right: gradient timing ────────────────────────────────────
ax = axes[1, 1]
grad_stages = ['Forward\npass', 'Gradient\n(warm-up)', 'Gradient\n(hot)']
grad_times = [t_fwd*1e3, t_grad_warmup*1e3, t_grad*1e3]
grad_colors = ['C0', 'C3', 'C2']
bars = ax.bar(grad_stages, grad_times, color=grad_colors, edgecolor='k', lw=0.5)
for bar, t in zip(bars, grad_times):
    label = f'{t:.0f} ms' if t < 1000 else f'{t/1e3:.1f} s'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(grad_times)*0.01,
            label, ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel("Time (ms)", fontsize=14)
ax.set_title("Differentiable pipeline timing", fontsize=14)
ax.set_xlabel("Stage", fontsize=14)

for ax in axes.flat:
    ax.tick_params(direction='in', which='both')
    ax.grid(alpha=0.15, ls='--')

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__) or '.', 'benchmark_cl_yy.png')
plt.savefig(outpath, dpi=150)
print(f"  Plot saved: {outpath}")

print("\n" + "=" * 65)
print("Done.")
print("=" * 65)
