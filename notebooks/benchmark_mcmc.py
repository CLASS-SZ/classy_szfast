"""
Benchmark: per-MCMC-step timing for C_ell^yy.

In an MCMC varying both cosmo + pressure profile parameters, every step
requires:
  1. build()            — CosmoPower emulators (P(k), H(z), chi, sigma)
  2. build_halo_grids() — sigma(M) interpolation, Tinker08 HMF, bias
  3. cl_yy_1h_2h()      — profile FT lookup, y_ell, 1h+2h integrals

This script times each stage separately, and also checks what happens
when we JIT-compile stage 3 (the only pure-JAX part).

Run:  python notebooks/benchmark_mcmc.py
"""
import warnings; warnings.filterwarnings('ignore')
import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"

import time
import numpy as np

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ── Import (this also builds FT tables) ──────────────────────────────
t_import_0 = time.time()
from classy_szfast.cosmology import build
from classy_szfast.hmf import build_halo_grids
from classy_szfast.power_spectrum import cl_yy_1h_2h
t_import = time.time() - t_import_0

# ── Setup ─────────────────────────────────────────────────────────────
cosmo = {
    'omega_b': 0.02242, 'omega_cdm': 0.11933, 'H0': 67.66,
    'tau_reio': 0.0561, 'ln10^{10}A_s': 3.047, 'n_s': 0.9665,
}
h = cosmo['H0'] / 100.0

z_grid = jnp.geomspace(0.005, 3.0, 100)
ell = jnp.geomspace(2, 8000, 50)

m_min = 5e10 / h   # M_sun
m_max = 3.5e15 / h

N_REPEAT = 10  # number of repetitions for averaging

print("=" * 65)
print("MCMC Benchmark: per-step timing for C_ell^yy")
print("=" * 65)
print(f"  n_z = {len(z_grid)},  n_ell = {len(ell)},  n_m = 300")
print(f"  Profile FT table import time: {t_import:.3f}s  (one-time cost)")
print()

# ── Stage 1: build() — emulator calls ────────────────────────────────
print("-" * 65)
print("Stage 1: build()  [emulator calls → CosmoGrids]")
print("-" * 65)

# Warmup
cg = build(cosmo, z_grid=z_grid)

times_build = []
for i in range(N_REPEAT):
    t0 = time.time()
    cg = build(cosmo, z_grid=z_grid)
    times_build.append(time.time() - t0)

t_build = np.median(times_build)
print(f"  median: {t_build*1e3:.1f} ms   (min={np.min(times_build)*1e3:.1f}, max={np.max(times_build)*1e3:.1f})")

# ── Stage 2: build_halo_grids() — HMF + bias ─────────────────────────
print("-" * 65)
print("Stage 2: build_halo_grids()  [sigma interp, HMF, bias → HaloGrids]")
print("-" * 65)

# Warmup
hg_500 = build_halo_grids(cg, cosmo, delta_crit=500.0, m_min=m_min, m_max=m_max, n_m=300)
hg_200 = build_halo_grids(cg, cosmo, delta_crit=200.0, m_min=m_min, m_max=m_max, n_m=300)

times_hg500 = []
times_hg200 = []
for i in range(N_REPEAT):
    t0 = time.time()
    hg_500 = build_halo_grids(cg, cosmo, delta_crit=500.0, m_min=m_min, m_max=m_max, n_m=300)
    times_hg500.append(time.time() - t0)

    t0 = time.time()
    hg_200 = build_halo_grids(cg, cosmo, delta_crit=200.0, m_min=m_min, m_max=m_max, n_m=300)
    times_hg200.append(time.time() - t0)

t_hg500 = np.median(times_hg500)
t_hg200 = np.median(times_hg200)
print(f"  Δ=500c: median {t_hg500*1e3:.1f} ms")
print(f"  Δ=200c: median {t_hg200*1e3:.1f} ms")

# ── Stage 3: cl_yy_1h_2h() — profile FT + integration ────────────────
print("-" * 65)
print("Stage 3: cl_yy_1h_2h()  [profile FT, y_ell, 1h+2h integrals]")
print("-" * 65)

# --- 3a. Without JIT (raw Python/JAX) ---
# Warmup
cl_1h, cl_2h = cl_yy_1h_2h(ell, cg, hg_500, cosmo, profile='arnaud10')

times_a10 = []
times_b12 = []
for i in range(N_REPEAT):
    t0 = time.time()
    cl_1h, cl_2h = cl_yy_1h_2h(ell, cg, hg_500, cosmo, profile='arnaud10')
    times_a10.append(time.time() - t0)

    t0 = time.time()
    cl_1h, cl_2h = cl_yy_1h_2h(ell, cg, hg_200, cosmo, profile='battaglia12')
    times_b12.append(time.time() - t0)

t_a10 = np.median(times_a10)
t_b12 = np.median(times_b12)
print(f"  No JIT:")
print(f"    A10:  median {t_a10*1e3:.1f} ms")
print(f"    B12:  median {t_b12*1e3:.1f} ms")

# --- 3b. With JIT ---
cl_yy_jit_a10 = jax.jit(lambda ell, cg, hg: cl_yy_1h_2h(ell, cg, hg, cosmo, profile='arnaud10'))
cl_yy_jit_b12 = jax.jit(lambda ell, cg, hg: cl_yy_1h_2h(ell, cg, hg, cosmo, profile='battaglia12'))

# JIT compile (first call)
t0 = time.time()
cl_1h_j, cl_2h_j = cl_yy_jit_a10(ell, cg, hg_500)
jax.block_until_ready(cl_1h_j)
t_jit_compile_a10 = time.time() - t0

t0 = time.time()
cl_1h_j, cl_2h_j = cl_yy_jit_b12(ell, cg, hg_200)
jax.block_until_ready(cl_1h_j)
t_jit_compile_b12 = time.time() - t0

print(f"\n  JIT compile (one-time):")
print(f"    A10:  {t_jit_compile_a10:.3f}s")
print(f"    B12:  {t_jit_compile_b12:.3f}s")

# JIT cached calls
times_a10_jit = []
times_b12_jit = []
for i in range(N_REPEAT):
    t0 = time.time()
    cl_1h_j, cl_2h_j = cl_yy_jit_a10(ell, cg, hg_500)
    jax.block_until_ready(cl_1h_j)
    times_a10_jit.append(time.time() - t0)

    t0 = time.time()
    cl_1h_j, cl_2h_j = cl_yy_jit_b12(ell, cg, hg_200)
    jax.block_until_ready(cl_1h_j)
    times_b12_jit.append(time.time() - t0)

t_a10_jit = np.median(times_a10_jit)
t_b12_jit = np.median(times_b12_jit)
print(f"\n  JIT cached:")
print(f"    A10:  median {t_a10_jit*1e3:.2f} ms")
print(f"    B12:  median {t_b12_jit*1e3:.2f} ms")

# ── Full MCMC step (no JIT on stage 3) ───────────────────────────────
print("\n" + "=" * 65)
print("Full MCMC step: build + halo_grids + cl_yy  (varying cosmo)")
print("=" * 65)

# Simulate different cosmo params
rng = np.random.default_rng(42)
cosmo_samples = []
for _ in range(N_REPEAT):
    c = dict(cosmo)
    c['omega_b']  = cosmo['omega_b']  * (1 + 0.01 * rng.standard_normal())
    c['omega_cdm'] = cosmo['omega_cdm'] * (1 + 0.01 * rng.standard_normal())
    c['H0']       = cosmo['H0']       * (1 + 0.01 * rng.standard_normal())
    c['ln10^{10}A_s'] = cosmo['ln10^{10}A_s'] * (1 + 0.01 * rng.standard_normal())
    c['n_s']      = cosmo['n_s']      * (1 + 0.01 * rng.standard_normal())
    cosmo_samples.append(c)

times_full_a10 = []
times_full_b12 = []
for c in cosmo_samples:
    h_c = c['H0'] / 100.0
    t0 = time.time()
    cg_c = build(c, z_grid=z_grid)
    hg_c = build_halo_grids(cg_c, c, delta_crit=500.0,
                             m_min=5e10/h_c, m_max=3.5e15/h_c, n_m=300)
    cl_1h, cl_2h = cl_yy_1h_2h(ell, cg_c, hg_c, c, profile='arnaud10')
    times_full_a10.append(time.time() - t0)

    t0 = time.time()
    cg_c = build(c, z_grid=z_grid)
    hg_c = build_halo_grids(cg_c, c, delta_crit=200.0,
                             m_min=5e10/h_c, m_max=3.5e15/h_c, n_m=300)
    cl_1h, cl_2h = cl_yy_1h_2h(ell, cg_c, hg_c, c, profile='battaglia12')
    times_full_b12.append(time.time() - t0)

t_full_a10 = np.median(times_full_a10)
t_full_b12 = np.median(times_full_b12)

print(f"  A10 (500c):  median {t_full_a10*1e3:.0f} ms/step")
print(f"  B12 (200c):  median {t_full_b12*1e3:.0f} ms/step")

# ── Breakdown summary ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Summary: per-step breakdown (median, ms)")
print("=" * 65)
print(f"  {'Stage':<35} {'Time (ms)':>10} {'% of total':>10}")
print(f"  {'-'*55}")

total = t_build + t_hg500 + t_a10
print(f"  {'build() [emulators]':<35} {t_build*1e3:>10.1f} {100*t_build/total:>9.0f}%")
print(f"  {'build_halo_grids() [HMF, bias]':<35} {t_hg500*1e3:>10.1f} {100*t_hg500/total:>9.0f}%")
print(f"  {'cl_yy_1h_2h() A10 [no JIT]':<35} {t_a10*1e3:>10.1f} {100*t_a10/total:>9.0f}%")
print(f"  {'─'*55}")
print(f"  {'Total (no JIT)':<35} {total*1e3:>10.1f}")
print(f"  {'cl_yy_1h_2h() A10 [JIT cached]':<35} {t_a10_jit*1e3:>10.2f}")
print(f"  {'Total (with JIT stage 3)':<35} {(t_build+t_hg500+t_a10_jit)*1e3:>10.1f}")
print()

# ── Profile FT table rebuild (for varying profile params) ────────────
print("=" * 65)
print("Profile FT table rebuild cost (if profile params vary)")
print("=" * 65)

from classy_szfast.power_spectrum import (
    _build_a10_ft_table, _build_b12_ft_table,
    _TABLE_U_GRID, _TABLE_SBT, _A10_GAMMA, _A10_ALPHA, _A10_BETA
)

times_a10_ft = []
times_b12_ft = []
for _ in range(N_REPEAT):
    t0 = time.time()
    _build_a10_ft_table()
    times_a10_ft.append(time.time() - t0)

    t0 = time.time()
    _build_b12_ft_table()
    times_b12_ft.append(time.time() - t0)

t_a10_ft = np.median(times_a10_ft)
t_b12_ft = np.median(times_b12_ft)
print(f"  A10 FT table (1D, {len(_TABLE_U_GRID)} pts):  median {t_a10_ft*1e3:.2f} ms")
print(f"  B12 FT table (2D, {len(_TABLE_U_GRID)}×100):  median {t_b12_ft*1e3:.1f} ms")
print(f"  → Negligible added cost per step")
print()

# ── Scaling with grid sizes ──────────────────────────────────────────
print("=" * 65)
print("Scaling: cl_yy_1h_2h() time vs grid size (A10, no JIT)")
print("=" * 65)

for nz, nm, nl in [(50, 100, 30), (100, 300, 50), (150, 300, 50), (200, 500, 80)]:
    z_g = jnp.geomspace(0.005, 3.0, nz)
    ell_g = jnp.geomspace(2, 8000, nl)
    cg_s = build(cosmo, z_grid=z_g)
    hg_s = build_halo_grids(cg_s, cosmo, delta_crit=500.0,
                              m_min=m_min, m_max=m_max, n_m=nm)
    # warmup
    cl_yy_1h_2h(ell_g, cg_s, hg_s, cosmo, profile='arnaud10')
    times = []
    for _ in range(5):
        t0 = time.time()
        cl_yy_1h_2h(ell_g, cg_s, hg_s, cosmo, profile='arnaud10')
        times.append(time.time() - t0)
    print(f"  n_z={nz:3d}  n_m={nm:3d}  n_ell={nl:2d}  →  {np.median(times)*1e3:.0f} ms")
