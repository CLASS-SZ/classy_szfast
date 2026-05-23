"""
Cosmological quantities from CosmoPower emulators (JAX).

This module provides the **grid-building layer**: call emulators once,
return a frozen ``CosmoGrids`` container that downstream JIT-compiled
functions can consume without any further Python overhead.

Unit system — *physical (h-free)* throughout
=============================================
Confirmed by comparison with CLASS Boltzmann (< 0.1 % agreement):

======== =============================== ==================
Quantity Symbol                          Unit
======== =============================== ==================
k        wavenumber                       1 / Mpc
P(k,z)   linear matter power spectrum     Mpc³
R        TophatVar smoothing radius       Mpc
sigma    rms density fluctuation          dimensionless
H(z)     Hubble rate / c                  1 / Mpc
chi(z)   comoving distance                Mpc
d_A(z)   angular diameter distance        Mpc
======== =============================== ==================

Masses, overdensities, and the R ↔ M relation are *not* handled here;
they belong in the halo-model layer where the overdensity definition
matters.

Design for JAX performance
==========================
* ``build(params, z_grid)`` calls the (non-JIT-able) emulators **once**
  and returns a ``CosmoGrids`` named-tuple.
* Every field of ``CosmoGrids`` is a plain ``jnp.array``.
* ``CosmoGrids`` is a JAX pytree (NamedTuple), so it can be passed
  directly into ``@jax.jit`` / ``jax.grad`` / ``jax.vmap`` functions.
* All downstream computation (HMF, profile FT, C_ell integration)
  should be written as **pure JAX functions of ``CosmoGrids``** — no
  emulator calls, no numpy, no Python loops.

CosmoPower PKL emulator internals
----------------------------------
The PKL emulator predicts  ``log10[ l(l+1) P(k) / (2π) ]``  where:

* *k* is sampled at ``geomspace(1e-4, 50, 5000)[::10]`` in **1/Mpc**
* *l* are artificial integer labels ``arange(2, 5002)[::10]``
* *P(k)* is the linear matter power spectrum in **Mpc³**

Recovery:  ``P(k) = 10^(emulator_output) / [l(l+1)/(2π)]``
"""

from __future__ import annotations

import warnings as _warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .cosmopower_jax import cp_pkl_nn_jax, cp_h_nn_jax, cp_da_nn_jax
from .emulators_meta_data import emulator_dict

jax.config.update("jax_enable_x64", True)


# ===================================================================
# Public data container
# ===================================================================

class CosmoGrids(NamedTuple):
    """Frozen container of precomputed cosmological arrays.

    Every field is a ``jnp.array``.  Being a ``NamedTuple`` makes this
    a JAX pytree, so it passes transparently through ``jax.jit``.

    Axes convention
    ---------------
    ========= ========================== ==============
    Field     Shape                       Unit
    ========= ========================== ==============
    z         (n_z,)                      —
    k         (n_k,)                      1 / Mpc
    pk        (n_z, n_k)                  Mpc³
    Hz        (n_z,)                      1 / Mpc
    chi       (n_z,)                      Mpc
    Da        (n_z,)                      Mpc
    R         (n_R,)                      Mpc
    sigma     (n_z, n_R)                  —
    dsigma2dR (n_z, n_R)                  1 / Mpc
    ========= ========================== ==============

    Notes
    -----
    * ``Hz`` is H(z)/c, not H(z).  Multiply by *c* = 299 792.458 km/s
      to get km s⁻¹ Mpc⁻¹.
    * ``R`` comes from TophatVar applied to physical-unit P(k), so it is
      in Mpc (not Mpc/h).
    * ``dsigma2dR`` = d(σ²)/dR, computed via ``np.gradient`` on the R grid.
    """
    z:         jax.Array   # (n_z,)
    k:         jax.Array   # (n_k,)
    pk:        jax.Array   # (n_z, n_k)
    Hz:        jax.Array   # (n_z,)
    chi:       jax.Array   # (n_z,)
    Da:        jax.Array   # (n_z,)
    R:         jax.Array   # (n_R,)
    sigma:     jax.Array   # (n_z, n_R)
    dsigma2dR: jax.Array   # (n_z, n_R)


# ===================================================================
# Builder — the only function that calls emulators
# ===================================================================

def build(params: dict,
          z_grid: jax.Array | None = None,
          n_z: int = 100,
          cosmo_model: str = 'lcdm') -> CosmoGrids:
    """Call emulators **once** and return a ``CosmoGrids`` container.

    Parameters
    ----------
    params : dict
        Cosmological parameters.  Must include at least
        ``omega_b``, ``omega_cdm``, ``H0``,
        ``ln10^{10}A_s``, ``n_s``.
    z_grid : 1-d array, optional
        Redshift grid.  Default: ``geomspace(5e-3, 5, n_z)``.
    n_z : int
        Number of z points when *z_grid* is not given.
    cosmo_model : str
        Cosmological model label for the emulators (default ``'lcdm'``).

    Returns
    -------
    CosmoGrids
        Named-tuple of JAX arrays, ready for downstream JIT functions.
    """
    from mcfit import TophatVar

    # Merge with emulator defaults
    full = dict(emulator_dict[cosmo_model]['default'])
    full.update(params)

    # --- z grid ---------------------------------------------------------
    if z_grid is None:
        z_grid = jnp.geomspace(5e-3, 5.0, n_z)
    z_grid = jnp.asarray(z_grid)
    z_grid = jnp.where(z_grid < 5e-3, 5e-3, z_grid)
    n_z_val = int(z_grid.shape[0])

    # --- P(k, z) --------------------------------------------------------
    k, pk = _predict_pk(full, z_grid, cosmo_model)     # (n_k,), (n_z, n_k)

    # --- H(z), chi(z), Da(z) -------------------------------------------
    Hz, chi, Da = _predict_distances(full, z_grid, cosmo_model)

    # --- sigma(R, z) and dsigma²/dR ------------------------------------
    R, sigma, dsigma2dR = _compute_sigma(k, pk, n_z_val)

    return CosmoGrids(
        z=z_grid, k=k, pk=pk,
        Hz=Hz, chi=chi, Da=Da,
        R=R, sigma=sigma, dsigma2dR=dsigma2dR,
    )


# ===================================================================
# Internal helpers (call emulators / mcfit)
# ===================================================================

# Per-cosmo_model PKL emulator grid + prefactor convention. Mirrors
# classy_szfast/classy_szfast.py (lines 99–118, 206–214, 267–275) which
# distinguishes 'ede-v2' from all other cosmologies.
#
# Final n_k = nk // ndspl. Final k = geomspace(kmin, kmax, nk)[::ndspl].
# prefac:
#   'ell' — emulator returns log10[ ell(ell+1) Pk / (2π) ] with ell labels
#           = arange(2, nk+2)[::ndspl]; recover Pk via × 1/(ell(ell+1)/(2π)).
#   'k3'  — emulator returns log10[ k³ Pk ]; recover Pk via × k^-3.
# extrap_kmin (optional): if set and < kmin, extend the Pk grid down to this
#   k using a primordial-slope extrapolation P(k) ∝ k^n_s. The matter Pk
#   on large scales (k << k_eq ≈ 0.01/Mpc) approaches this asymptotic shape;
#   the extrapolation gives ~1% accuracy down to k ~ 1e-4/Mpc.
_PK_GRID_CONFIG = {
    'ede-v2': dict(kmin=5e-4, kmax=10.0,  nk=1000, ndspl=1,  prefac='k3',
                   extrap_kmin=1e-4),
    # Default (everything else: 'lcdm', 'mnu', 'neff', 'wcdm', 'ede', 'mnu-3states')
    '_default': dict(kmin=1e-4, kmax=50.0, nk=5000, ndspl=10, prefac='ell',
                     extrap_kmin=None),
}

def _predict_pk(full: dict, z_grid: jax.Array,
                cosmo_model: str):
    """P(k, z): k in 1/Mpc, P in Mpc³.

    For cosmo_models whose emulator has a higher kmin than the lcdm default,
    optionally extend down via P(k) ∝ k^n_s (primordial slope, valid for
    k << k_eq). The extra points use the SAME log-k step as the emulator
    grid so the combined array stays uniformly log-spaced — required by
    downstream mcfit (TophatVar) σ(R) integration.
    See ``_PK_GRID_CONFIG[…]['extrap_kmin']``.
    """
    cfg = _PK_GRID_CONFIG.get(cosmo_model, _PK_GRID_CONFIG['_default'])

    # k grid (1/Mpc) — emulator native
    k_emu = jnp.geomspace(cfg['kmin'], cfg['kmax'], cfg['nk'])[::cfg['ndspl']]
    n_k_emu = k_emu.shape[0]

    # Prefactor used to undo the emulator's normalisation
    if cfg['prefac'] == 'ell':
        ls = jnp.arange(2, cfg['nk'] + 2)[::cfg['ndspl']]
        pk_power_fac = 1.0 / (ls * (ls + 1.0) / (2.0 * jnp.pi))
    else:                                       # 'k3'
        pk_power_fac = k_emu ** -3

    # Batched parameter dict
    n_z = int(z_grid.shape[0])
    params_pk = {k_name: [v] * n_z for k_name, v in full.items()}
    params_pk['z_pk_save_nonclass'] = list(z_grid)

    # Emulator: log10[ prefactor × P(k) ]
    log10pk = cp_pkl_nn_jax[cosmo_model].predict(params_pk)

    pk_emu = jnp.float64(10.0**log10pk * pk_power_fac)       # (n_z, n_k_emu)

    # Optional low-k extrapolation with P(k) ∝ k^n_s
    extrap_kmin = cfg.get('extrap_kmin')
    if extrap_kmin is not None and extrap_kmin < cfg['kmin']:
        n_s = full['n_s']
        # Match the emulator's log-step so the combined array stays uniformly
        # log-spaced (mcfit requires this for the TophatVar σ(R) integration).
        log10_step = (float(jnp.log10(cfg['kmax'])) - float(jnp.log10(cfg['kmin']))) / (cfg['nk'] - 1)
        # Account for emulator's stride (ndspl) — log-step on the *output* grid
        log10_step_out = log10_step * cfg['ndspl']
        # Number of extra points to bridge from extrap_kmin to kmin at this step
        n_extra = int(round((float(jnp.log10(cfg['kmin'])) - float(jnp.log10(extrap_kmin)))
                            / log10_step_out))
        # Build extrapolation k-grid with the same log-step; exclude kmin (already in k_emu)
        log10_k_extra = jnp.log10(cfg['kmin']) - jnp.arange(n_extra, 0, -1) * log10_step_out
        k_extra = jnp.power(10.0, log10_k_extra)
        pk_at_kmin = pk_emu[:, 0:1]                          # (n_z, 1)
        pk_extra = pk_at_kmin * (k_extra[None, :] / cfg['kmin']) ** n_s
        k  = jnp.concatenate([k_extra, k_emu])
        pk = jnp.concatenate([pk_extra, pk_emu], axis=1)
    else:
        k = k_emu
        pk = pk_emu

    return k, pk


def _predict_distances(full: dict, z_grid: jax.Array,
                       cosmo_model: str):
    """H(z)/c (1/Mpc), chi (Mpc), Da (Mpc) interpolated to z_grid.

    Both HZ and DAZ emulators were trained on z ∈ [0, 20] sampled by
    ``linspace(0, 20, 5000)`` (cp_z_interp in classy_szfast.py:279-280).
    For lcdm-style cosmologies the DAZ emulator includes z=0 (returning
    chi(0)=Da(0)=0 as the first point); for ede-v2 the DAZ emulator was
    trained without the z=0 anchor and returns only 4999 points starting
    at z=0.004, so we prepend a zero — same fix as
    classy_szfast.py:1110-1111 in the cobaya wrapper. Without this,
    chi(z=0.005) for ede-v2 is wrong by ~80%.
    """
    params_one = {k: [v] for k, v in full.items()}

    Hz_fine = cp_h_nn_jax[cosmo_model].predict(params_one)   # 1/Mpc
    Da_fine = cp_da_nn_jax[cosmo_model].predict(params_one)  # Mpc

    # ede-v2 DAZ emulator omits z=0; restore it so chi(0)=0 and the
    # underlying z-grid matches HZ. Mirrors classy_szfast.py:1110.
    if cosmo_model == 'ede-v2':
        Da_fine = jnp.concatenate(
            [jnp.zeros(1, dtype=Da_fine.dtype), Da_fine])

    # Both H and Da emulators share the same z-grid after the z=0 fix above.
    zmax_dist = 20.0
    z_fine_h = jnp.linspace(0.0, zmax_dist, Hz_fine.shape[-1])
    z_fine_d = jnp.linspace(0.0, zmax_dist, Da_fine.shape[-1])

    chi_fine = Da_fine * (1.0 + z_fine_d)                    # Mpc

    Hz  = jnp.interp(z_grid, z_fine_h, Hz_fine)
    chi = jnp.interp(z_grid, z_fine_d, chi_fine)
    chi = jnp.where(chi < 1.0, 1.0, chi)
    Da  = jnp.interp(z_grid, z_fine_d, Da_fine)

    return Hz, chi, Da


def _compute_sigma(k: jax.Array, pk: jax.Array, n_z: int):
    """sigma(R, z) and dsigma²/dR from TophatVar.

    Returns R (Mpc), sigma (n_z, n_R), dsigma2dR (n_z, n_R).
    """
    from mcfit import TophatVar

    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore",
                                 message="use backend='jax' if desired")
        tv = TophatVar(np.array(k, copy=False), lowring=True, backend='jax')

    R_all, var_all = tv(pk, extrap=True)            # (n_z, n_R)
    R = R_all.flatten()                              # shared across z

    sigma = jnp.sqrt(var_all)

    # dsigma²/dR via central finite differences (pure JAX, differentiable)
    dsigma2dR = jax.vmap(lambda var_z: jnp.gradient(var_z, R))(var_all)

    return R, sigma, dsigma2dR


# ===================================================================
# Convenience accessors (for standalone use / notebooks)
# ===================================================================

def get_pk(params, z_arr, cosmo_model='lcdm'):
    """Quick accessor: P(k, z).  k in 1/Mpc, P in Mpc³."""
    full = dict(emulator_dict[cosmo_model]['default'])
    full.update(params)
    return _predict_pk(full, jnp.asarray(z_arr), cosmo_model)


def get_distances(params, z_arr, cosmo_model='lcdm'):
    """Quick accessor: Hz (1/Mpc), chi (Mpc), Da (Mpc)."""
    full = dict(emulator_dict[cosmo_model]['default'])
    full.update(params)
    return _predict_distances(full, jnp.asarray(z_arr), cosmo_model)


def get_sigma(params, z_arr, cosmo_model='lcdm'):
    """Quick accessor: R (Mpc), sigma, dsigma²/dR (1/Mpc)."""
    k, pk = get_pk(params, z_arr, cosmo_model)
    n_z = len(z_arr)
    return _compute_sigma(k, pk, n_z)
