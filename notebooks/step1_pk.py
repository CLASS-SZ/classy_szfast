"""
Step 1: Compare P(k) from our JAX emulator vs CLASS-SZ Boltzmann.

Three curves per redshift:
  1. Our emulator  (cosmology.get_pk_phys)
  2. CLASS-SZ fast mode  (csz.pk_lin in fast mode)
  3. CLASS Boltzmann      (csz.pk_lin in compute mode)

Units: k in 1/Mpc, P(k) in Mpc^3.

Run:  python notebooks/step1_pk.py
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

# ── 1. Our JAX emulator ─────────────────────────────────────────────────────
print("=" * 60)
print("1. JAX emulator (cosmology.get_pk_phys)")
print("=" * 60)

from classy_szfast.cosmology import get_pk

t0 = time.time()
k_phys, pk_phys = get_pk(cosmo, jnp.array(z_test))
t_emu = time.time() - t0

k_np = np.array(k_phys)
pk_np = np.array(pk_phys)  # shape (n_z, n_k)

print(f"  k range: [{k_np.min():.2e}, {k_np.max():.2e}] 1/Mpc")
print(f"  n_k = {len(k_np)},  n_z = {len(z_test)}")
print(f"  elapsed = {t_emu:.2f}s")
for iz, z in enumerate(z_test):
    print(f"  z={z:.1f}: P(k=0.1/Mpc) ≈ {np.interp(np.log(0.1), np.log(k_np), np.log(pk_np[iz])):.2f} (ln)")

# ── 2. CLASS-SZ Boltzmann ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CLASS-SZ Boltzmann (full compute)")
print("=" * 60)

from classy_sz import Class as Class_sz

t0 = time.time()
csz_boltz = Class_sz()
csz_boltz.set({
    **cosmo,
    'output': 'mPk',
    'P_k_max_1/Mpc': 50.0 * h,   # ensure we cover k range
    'z_max_pk': 3.0,
    # Ensure non-fast mode: do NOT call initialize_classy_szfast
})
csz_boltz.compute()
t_boltz = time.time() - t0
print(f"  elapsed = {t_boltz:.1f}s")

# Evaluate at our k grid
pk_boltz = np.zeros((len(z_test), len(k_np)))
for iz, z in enumerate(z_test):
    for ik, kv in enumerate(k_np):
        try:
            pk_boltz[iz, ik] = csz_boltz.pk_lin(kv, z)
        except Exception:
            pk_boltz[iz, ik] = np.nan

print(f"  P(k) units: Mpc^3 (documented in classy.pyx)")
for iz, z in enumerate(z_test):
    valid = ~np.isnan(pk_boltz[iz])
    if valid.any():
        print(f"  z={z:.1f}: P(k=0.1) = {np.interp(np.log(0.1), np.log(k_np[valid]), np.log(pk_boltz[iz, valid])):.2f} (ln)")

# ── 3. CLASS-SZ fast mode ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. CLASS-SZ fast mode")
print("=" * 60)

t0 = time.time()
csz_fast = Class_sz()
csz_fast.set({
    **cosmo,
    'skip_input': 0,
    'output': 'tSZ_1h',
    'cosmo_model': 0,
})
csz_fast.initialize_classy_szfast()
t_fast = time.time() - t0
print(f"  init elapsed = {t_fast:.1f}s")

# pk_lin(k, z) — k in 1/Mpc, returns Mpc^3
pk_fast = np.zeros((len(z_test), len(k_np)))
for iz, z in enumerate(z_test):
    for ik, kv in enumerate(k_np):
        try:
            pk_fast[iz, ik] = csz_fast.pk_lin(float(kv), float(z))
        except Exception:
            pk_fast[iz, ik] = np.nan

for iz, z in enumerate(z_test):
    valid = ~np.isnan(pk_fast[iz]) & (pk_fast[iz] > 0)
    if valid.any():
        print(f"  z={z:.1f}: P(k=0.1) = {np.interp(np.log(0.1), np.log(k_np[valid]), np.log(pk_fast[iz, valid])):.2f} (ln)")

# ── 4. Print comparison table ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Ratios at selected k values")
print("=" * 60)

k_check = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
for z in z_test:
    iz = z_test.index(z)
    print(f"\n  z = {z:.1f}:")
    print(f"  {'k [1/Mpc]':>12}  {'emu/boltz':>10}  {'fast/boltz':>10}  {'emu/fast':>10}")
    for kv in k_check:
        if kv > k_np.max() or kv < k_np.min():
            continue
        pk_e = np.interp(np.log(kv), np.log(k_np), np.log(pk_np[iz]))
        pk_e = np.exp(pk_e)

        valid_b = ~np.isnan(pk_boltz[iz]) & (pk_boltz[iz] > 0)
        pk_b = np.interp(np.log(kv), np.log(k_np[valid_b]), np.log(pk_boltz[iz, valid_b]))
        pk_b = np.exp(pk_b)

        valid_f = ~np.isnan(pk_fast[iz]) & (pk_fast[iz] > 0)
        if valid_f.any():
            pk_f = np.interp(np.log(kv), np.log(k_np[valid_f]), np.log(pk_fast[iz, valid_f]))
            pk_f = np.exp(pk_f)
        else:
            pk_f = np.nan

        r_eb = pk_e / pk_b if pk_b > 0 else np.nan
        r_fb = pk_f / pk_b if (pk_b > 0 and not np.isnan(pk_f)) else np.nan
        r_ef = pk_e / pk_f if (not np.isnan(pk_f) and pk_f > 0) else np.nan
        print(f"  {kv:12.3f}  {r_eb:10.4f}  {r_fb:10.4f}  {r_ef:10.4f}")

# ── 5. Plot ──────────────────────────────────────────────────────────────────
label_size = 17
title_size = 18
legend_size = 12
handle_length = 1.5

fig, axes = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [3, 1]},
                          sharex=True)
ax_pk, ax_rat = axes

for ax in axes:
    ax.tick_params(axis='x', which='both', length=5, direction='in', pad=10)
    ax.tick_params(axis='y', which='both', length=5, direction='in', pad=5)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=label_size)
    plt.setp(ax.get_xticklabels(), fontsize=label_size)
    ax.grid(visible=True, which='both', alpha=0.2, linestyle='--')

colors = ['C0', 'C1', 'C2', 'C3']
for iz, z in enumerate(z_test):
    c = colors[iz]
    valid_b = ~np.isnan(pk_boltz[iz]) & (pk_boltz[iz] > 0)
    valid_f = ~np.isnan(pk_fast[iz]) & (pk_fast[iz] > 0)

    # P(k) panel
    ax_pk.loglog(k_np, pk_np[iz], color=c, ls='-', lw=2,
                 label=f'emulator z={z}')
    if valid_b.any():
        ax_pk.loglog(k_np[valid_b], pk_boltz[iz, valid_b], color=c,
                     ls='--', lw=1.5)
    if valid_f.any():
        ax_pk.loglog(k_np[valid_f], pk_fast[iz, valid_f], color=c,
                     ls=':', lw=1.5)

    # Ratio panel (emulator / Boltzmann)
    if valid_b.any():
        ratio_eb = pk_np[iz, valid_b] / pk_boltz[iz, valid_b]
        ax_rat.semilogx(k_np[valid_b], ratio_eb, color=c, ls='-', lw=2,
                        label=f'emu/boltz z={z}')
    if valid_f.any() and valid_b.any():
        # fast / Boltzmann
        both = valid_f & valid_b
        if both.any():
            ratio_fb = pk_fast[iz, both] / pk_boltz[iz, both]
            ax_rat.semilogx(k_np[both], ratio_fb, color=c, ls=':', lw=1.5)

ax_pk.set_ylabel(r"$P_{\rm lin}(k)\;[\mathrm{Mpc}^3]$", size=title_size)
ax_pk.legend(fontsize=legend_size-1, handlelength=handle_length, ncol=2)
ax_pk.set_title("Solid=emulator, Dashed=Boltzmann, Dotted=fast", fontsize=13)

ax_rat.set_xlabel(r"$k\;[1/\mathrm{Mpc}]$", size=title_size)
ax_rat.set_ylabel("ratio", size=title_size)
ax_rat.axhline(1.0, color='k', ls='-', lw=0.5)
ax_rat.set_ylim(0.95, 1.05)
ax_rat.legend(fontsize=legend_size-2, handlelength=handle_length, ncol=2)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__) or '.', 'step1_pk.png')
plt.savefig(outpath, dpi=150)
print(f"\nPlot saved: {outpath}")

# Cleanup
csz_boltz.struct_cleanup()
csz_fast.struct_cleanup()
