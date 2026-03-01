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

def _predict_pk(full: dict, z_grid: jax.Array,
                cosmo_model: str):
    """P(k, z): k in 1/Mpc, P in Mpc³."""
    # k grid (1/Mpc) — emulator native
    k = jnp.geomspace(1e-4, 50.0, 5000)[::10]          # 500 pts

    # Artificial ell labels for the power-factor
    ls = jnp.arange(2, 5002)[::10]
    pk_power_fac = 1.0 / (ls * (ls + 1.0) / (2.0 * jnp.pi))

    # Batched parameter dict
    n_z = int(z_grid.shape[0])
    params_pk = {k_name: [v] * n_z for k_name, v in full.items()}
    params_pk['z_pk_save_nonclass'] = [float(z) for z in z_grid]

    # Emulator: log10[ l(l+1) P(k) / (2π) ]
    log10pk = cp_pkl_nn_jax[cosmo_model].predict(params_pk)

    pk = jnp.float64(10.0**log10pk * pk_power_fac)       # (n_z, n_k)
    return k, pk


def _predict_distances(full: dict, z_grid: jax.Array,
                       cosmo_model: str):
    """H(z)/c (1/Mpc), chi (Mpc), Da (Mpc) interpolated to z_grid."""
    z_fine = jnp.linspace(0.0, 20.0, 5000)
    params_one = {k: [v] for k, v in full.items()}

    Hz_fine = cp_h_nn_jax[cosmo_model].predict(params_one)   # 1/Mpc
    Da_fine = cp_da_nn_jax[cosmo_model].predict(params_one)  # Mpc
    chi_fine = Da_fine * (1.0 + z_fine)                      # Mpc

    Hz  = jnp.interp(z_grid, z_fine, Hz_fine)
    chi = jnp.interp(z_grid, z_fine, chi_fine)
    chi = jnp.where(chi < 1.0, 1.0, chi)
    Da  = jnp.interp(z_grid, z_fine, Da_fine)

    return Hz, chi, Da


def _compute_sigma(k: jax.Array, pk: jax.Array, n_z: int):
    """sigma(R, z) and dsigma²/dR from TophatVar.

    Returns R (Mpc), sigma (n_z, n_R), dsigma2dR (n_z, n_R).
    """
    from mcfit import TophatVar

    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore",
                                 message="use backend='jax' if desired")
        tv = TophatVar(np.asarray(k), lowring=True, backend='jax')

    R_all, var_all = tv(pk, extrap=True)            # (n_z, n_R)
    R = R_all.flatten()                              # shared across z

    sigma = jnp.sqrt(var_all)

    # dsigma²/dR via central finite differences (numpy, run once)
    R_np  = np.asarray(R)
    var_np = np.asarray(var_all)
    dvar_np = np.empty_like(var_np)
    for iz in range(n_z):
        dvar_np[iz, :] = np.gradient(var_np[iz, :], R_np)
    dsigma2dR = jnp.array(dvar_np)

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
