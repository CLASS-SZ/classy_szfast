"""
Step 5: Compare C_ell^yy from our JAX pipeline vs CLASS-SZ (run live).

Follows the GNFW calculation from:
  class_sz/docs/notebooks/class_sz_tszpowerspectrum.ipynb

Uses CLASS-SZ fast mode (initialize_classy_szfast) for the benchmark.

Run:  python notebooks/step5_cl_yy.py
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

# ── Cosmology (matches notebook) ─────────────────────────────────────────────
cosmo = {
    'omega_b': 0.02, 'omega_cdm': 0.12, 'H0': 80.,
    'tau_reio': 0.0561, 'ln10^{10}A_s': 3.047, 'n_s': 0.9665,
}
h = cosmo['H0'] / 100.0

# Integration limits (CLASS-SZ convention: masses in M_sun/h)
z_min, z_max = 0.005, 2.0
M_min_h, M_max_h = 5e10, 3.5e15  # M_sun/h

# Convert to physical M_sun for our code
m_min = M_min_h / h
m_max = M_max_h / h

# ── 1. Run CLASS-SZ live ────────────────────────────────────────────────────
print("=" * 65)
print("1. CLASS-SZ (GNFW profile, T08M500c, fast mode)")
print("=" * 65)

from classy_sz import Class as Class_sz

t0 = time.time()
csz = Class_sz()
csz.set(cosmo)
csz.set({
    'output': 'tSZ_tSZ_1h,tSZ_tSZ_2h',
    'skip_input': 0,
    'cosmo_model': 0,

    'ell_min': 2,
    'ell_max': 8000,
    'dell': 0.,
    'dlogell': 0.2,

    'z_min': z_min,
    'z_max': z_max,
    'M_min': M_min_h,
    'M_max': M_max_h,

    'mass_function': 'T08M500c',
    'pressure_profile': 'GNFW',

    'P0GNFW': 8.130,
    'c500': 1.156,
    'gammaGNFW': 0.3292,
    'alphaGNFW': 1.0620,
    'betaGNFW': 5.4807,

    # B=1.0 means no mass bias (our code has no B parameter yet)
    'B': 1.0,

    'no_spline_in_tinker': 1,
    'HMF_prescription_NCDM': 1,
    'use_resnet_pkl': 0,

    # Precision
    'x_outSZ': 4.,
    'use_fft_for_profiles_transform': 1,
    'N_samp_fftw': 1024,
    'x_min_gas_pressure_fftw': 1e-4,
    'x_max_gas_pressure_fftw': 1e6,
    'n_m_pressure_profile': 100,
    'n_z_pressure_profile': 100,
    'ndim_redshifts': 150,
    'redshift_epsabs': 1e-40,
    'redshift_epsrel': 1e-4,
    'mass_epsabs': 1e-40,
    'mass_epsrel': 1e-4,
})
csz.initialize_classy_szfast()
t_csz = time.time() - t0

ell_csz = np.asarray(csz.cl_sz()['ell'])
dl_1h_csz = np.asarray(csz.cl_sz()['1h'])
dl_2h_csz = np.asarray(csz.cl_sz()['2h'])

print(f"  elapsed: {t_csz:.2f}s")
print(f"  ell range: [{ell_csz[0]:.0f}, {ell_csz[-1]:.0f}], n_ell={len(ell_csz)}")
print(f"  1h range: [{dl_1h_csz.min():.4e}, {dl_1h_csz.max():.4e}]")
print(f"  2h range: [{dl_2h_csz.min():.4e}, {dl_2h_csz.max():.4e}]")

# ── 2. Our JAX pipeline ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("2. JAX pipeline (cosmology.build + hmf + power_spectrum)")
print("=" * 65)

from classy_szfast.cosmology import build
from classy_szfast.hmf import build_halo_grids
from classy_szfast.power_spectrum import cl_yy_1h_2h

ell_jax = jnp.array(ell_csz)

t0 = time.time()
z_grid = jnp.geomspace(z_min, z_max, 150)
cg = build(cosmo, z_grid=z_grid)
hg = build_halo_grids(cg, cosmo, delta_crit=500.0,
                       m_min=m_min, m_max=m_max, n_m=300)
cl_1h, cl_2h = cl_yy_1h_2h(ell_jax, cg, hg, cosmo, profile='arnaud10')
t_jax = time.time() - t0

dl_fac = ell_jax * (ell_jax + 1) / (2 * jnp.pi) * 1e12
dl_1h_jax = np.array(dl_fac * cl_1h)
dl_2h_jax = np.array(dl_fac * cl_2h)

print(f"  elapsed: {t_jax:.2f}s")
print(f"  1h range: [{dl_1h_jax.min():.4e}, {dl_1h_jax.max():.4e}]")
print(f"  2h range: [{dl_2h_jax.min():.4e}, {dl_2h_jax.max():.4e}]")

# ── 3. Comparison ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Ratios JAX / CLASS-SZ")
print("=" * 65)

r1h = dl_1h_jax / dl_1h_csz
r2h = dl_2h_jax / dl_2h_csz
print(f"  1h: mean={r1h.mean():.4f}  min={r1h.min():.4f}  max={r1h.max():.4f}")
print(f"  2h: mean={r2h.mean():.4f}  min={r2h.min():.4f}  max={r2h.max():.4f}")

print(f"\n  {'ell':>6}  {'JAX 1h':>10}  {'CSZ 1h':>10}  {'1h ratio':>8}"
      f"  {'JAX 2h':>10}  {'CSZ 2h':>10}  {'2h ratio':>8}")
for i in range(0, len(ell_csz), max(1, len(ell_csz)//10)):
    e = float(ell_csz[i])
    print(f"  {e:6.0f}  {dl_1h_jax[i]:10.4e}  {dl_1h_csz[i]:10.4e}  {r1h[i]:8.4f}"
          f"  {dl_2h_jax[i]:10.4e}  {dl_2h_csz[i]:10.4e}  {r2h[i]:8.4f}")

# ── 4. Also run B12 comparison ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("3. CLASS-SZ B12 (for additional comparison)")
print("=" * 65)

csz_b12 = Class_sz()
csz_b12.set(cosmo)
csz_b12.set({
    'output': 'tSZ_1h,tSZ_2h',
    'skip_input': 0,
    'cosmo_model': 0,
    'ell_min': 2, 'ell_max': 8000,
    'dell': 0., 'dlogell': 0.2,
    'z_min': z_min, 'z_max': z_max,
    'M_min': M_min_h, 'M_max': M_max_h,
    'mass_function': 'T08M200c',
    'pressure_profile': 'B12',
    'concentration_parameter': 'D08',
    'x_outSZ': 4.,
    'use_fft_for_profiles_transform': 1,
    'N_samp_fftw': 1024,
    'x_min_gas_pressure_fftw': 1e-4,
    'x_max_gas_pressure_fftw': 1e6,
    'ndim_redshifts': 150,
    'n_m_pressure_profile': 100,
    'n_z_pressure_profile': 100,
})
csz_b12.compute_class_szfast()

ell_b12 = np.asarray(csz_b12.cl_sz()['ell'])
dl_1h_csz_b12 = np.asarray(csz_b12.cl_sz()['1h'])
dl_2h_csz_b12 = np.asarray(csz_b12.cl_sz()['2h'])

# Our B12
hg_200c = build_halo_grids(cg, cosmo, delta_crit=200.0,
                            m_min=m_min, m_max=m_max, n_m=300)
cl_1h_b12, cl_2h_b12 = cl_yy_1h_2h(jnp.array(ell_b12), cg, hg_200c, cosmo,
                                      profile='battaglia12')
dl_fac_b12 = jnp.array(ell_b12) * (jnp.array(ell_b12) + 1) / (2 * jnp.pi) * 1e12
dl_1h_jax_b12 = np.array(dl_fac_b12 * cl_1h_b12)
dl_2h_jax_b12 = np.array(dl_fac_b12 * cl_2h_b12)

r1h_b12 = dl_1h_jax_b12 / dl_1h_csz_b12
r2h_b12 = dl_2h_jax_b12 / dl_2h_csz_b12
print(f"  B12 1h: mean={r1h_b12.mean():.4f}  [{r1h_b12.min():.4f}, {r1h_b12.max():.4f}]")
print(f"  B12 2h: mean={r2h_b12.mean():.4f}  [{r2h_b12.min():.4f}, {r2h_b12.max():.4f}]")

# ── 5. Plot ─────────────────────────────────────────────────────────────────
label_size = 17
title_size = 18
legend_size = 11

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Top left: GNFW (A10) D_ell
ax = axes[0, 0]
ax.loglog(ell_csz, dl_1h_jax, 'C0-', lw=2, label='JAX 1h')
ax.loglog(ell_csz, dl_2h_jax, 'C1-', lw=2, label='JAX 2h')
ax.loglog(ell_csz, dl_1h_jax + dl_2h_jax, 'k-', lw=2, label='JAX total')
ax.loglog(ell_csz, dl_1h_csz, 'C0--', lw=1.5, label='CLASS-SZ 1h')
ax.loglog(ell_csz, dl_2h_csz, 'C1--', lw=1.5, label='CLASS-SZ 2h')
ax.set_ylabel(r"$D_\ell^{yy} \times 10^{12}$", size=title_size)
ax.set_title("GNFW (A10 params), T08M500c", fontsize=14)
ax.legend(fontsize=legend_size - 1, ncol=2)

# Top right: GNFW ratio
ax = axes[0, 1]
ax.semilogx(ell_csz, r1h, 'C0-', lw=2, label='1h')
ax.semilogx(ell_csz, r2h, 'C1-', lw=2, label='2h')
ax.axhline(1.0, color='k', ls=':', lw=0.5)
ax.set_ylabel("JAX / CLASS-SZ", size=title_size)
ax.set_title("GNFW ratio", fontsize=14)
ax.legend(fontsize=legend_size)

# Bottom left: B12 D_ell
ax = axes[1, 0]
ax.loglog(ell_b12, dl_1h_jax_b12, 'C0-', lw=2, label='JAX 1h')
ax.loglog(ell_b12, dl_2h_jax_b12, 'C1-', lw=2, label='JAX 2h')
ax.loglog(ell_b12, dl_1h_csz_b12, 'C0--', lw=1.5, label='CLASS-SZ 1h')
ax.loglog(ell_b12, dl_2h_csz_b12, 'C1--', lw=1.5, label='CLASS-SZ 2h')
ax.set_xlabel(r"$\ell$", size=title_size)
ax.set_ylabel(r"$D_\ell^{yy} \times 10^{12}$", size=title_size)
ax.set_title("Battaglia 2012, T08M200c", fontsize=14)
ax.legend(fontsize=legend_size - 1, ncol=2)

# Bottom right: B12 ratio
ax = axes[1, 1]
ax.semilogx(ell_b12, r1h_b12, 'C0-', lw=2, label='1h')
ax.semilogx(ell_b12, r2h_b12, 'C1-', lw=2, label='2h')
ax.axhline(1.0, color='k', ls=':', lw=0.5)
ax.set_xlabel(r"$\ell$", size=title_size)
ax.set_ylabel("JAX / CLASS-SZ", size=title_size)
ax.set_title("B12 ratio", fontsize=14)
ax.legend(fontsize=legend_size)

for ax in axes.flat:
    ax.tick_params(axis='x', which='both', length=5, direction='in', pad=10)
    ax.tick_params(axis='y', which='both', length=5, direction='in', pad=5)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.setp(ax.get_yticklabels(), fontsize=label_size - 3)
    plt.setp(ax.get_xticklabels(), fontsize=label_size - 3)
    ax.grid(visible=True, which='both', alpha=0.2, linestyle='--')

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__) or '.', 'step5_cl_yy.png')
plt.savefig(outpath, dpi=150)
print(f"\nPlot saved: {outpath}")

csz.struct_cleanup()
csz_b12.struct_cleanup()
