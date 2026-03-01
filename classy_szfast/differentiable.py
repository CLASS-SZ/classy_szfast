"""
End-to-end differentiable C_ell^yy pipeline.

Provides ``cl_yy_from_params`` — a single function that takes
cosmological and (optionally) pressure-profile parameters and returns
the 1-halo and 2-halo tSZ angular power spectra, compatible with
``jax.grad``, ``jax.jacfwd``, and ``jax.jit``.

Usage
-----
>>> from classy_szfast.differentiable import (
...     CosmoParams, ProfileParamsA10, cl_yy_from_params)
>>>
>>> cosmo = CosmoParams(omega_b=0.02242, omega_cdm=0.11933,
...                     H0=67.66, tau_reio=0.0561,
...                     ln10_10_As=3.047, n_s=0.9665)
>>> ell = jnp.arange(2, 3001, dtype=jnp.float64)
>>> cl_1h, cl_2h = cl_yy_from_params(ell, cosmo)
>>>
>>> # Gradient w.r.t. omega_b
>>> def loss(omega_b):
...     c = cosmo._replace(omega_b=omega_b)
...     cl_1h, cl_2h = cl_yy_from_params(ell, c)
...     return jnp.sum(cl_1h + cl_2h)
>>> grad_omega_b = jax.grad(loss)(0.02242)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cosmology import build as build_cosmo_grids
from .hmf import build_halo_grids
from .power_spectrum import cl_yy_1h_2h

jax.config.update("jax_enable_x64", True)


# ===================================================================
# Parameter containers (JAX pytrees via NamedTuple)
# ===================================================================

class CosmoParams(NamedTuple):
    """Cosmological parameters for the emulator pipeline.

    All fields accept ``float`` or ``jax.Array`` — the latter enables
    differentiation via ``jax.grad``.
    """
    omega_b:      float | jax.Array
    omega_cdm:    float | jax.Array
    H0:           float | jax.Array
    tau_reio:     float | jax.Array
    ln10_10_As:   float | jax.Array
    n_s:          float | jax.Array
    m_ncdm:       float | jax.Array = 0.06


class ProfileParamsA10(NamedTuple):
    """Arnaud 2010 gNFW pressure-profile parameters.

    Defaults are the Arnaud et al. 2010 best-fit values.
    """
    P0:    float | jax.Array = 8.130
    c500:  float | jax.Array = 1.156
    gamma: float | jax.Array = 0.3292
    alpha: float | jax.Array = 1.0620
    beta:  float | jax.Array = 5.4807


class ProfileParamsB12(NamedTuple):
    """Battaglia 2012 gNFW fitting coefficients.

    Each of P0, xc, beta follows:  A × (M/10¹⁴)^α_m × (1+z)^α_z.
    Defaults are the Battaglia et al. 2012 Table 1 best-fit values.
    """
    P0_A:      float | jax.Array = 18.1
    P0_am:     float | jax.Array = 0.154
    P0_az:     float | jax.Array = -0.758
    xc_A:      float | jax.Array = 0.497
    xc_am:     float | jax.Array = -0.00865
    xc_az:     float | jax.Array = 0.731
    beta_A:    float | jax.Array = 4.35
    beta_am:   float | jax.Array = 0.0393
    beta_az:   float | jax.Array = 0.415


# ===================================================================
# End-to-end differentiable pipeline
# ===================================================================

def cl_yy_from_params(
    ell: jax.Array,
    cosmo: CosmoParams,
    profile_params: ProfileParamsA10 | ProfileParamsB12 | None = None,
    *,
    profile: str = 'arnaud10',
    z_grid: jax.Array | None = None,
    n_z: int = 100,
    cosmo_model: str = 'lcdm',
    delta_crit: float = 500.0,
    m_min: float = 1e10,
    m_max: float = 5e16,
    n_m: int = 200,
) -> tuple[jax.Array, jax.Array]:
    """Compute C_ell^yy from cosmological (and profile) parameters.

    This function chains the three pipeline stages:

    1. ``build()``            — emulators → ``CosmoGrids``
    2. ``build_halo_grids()`` — HMF/bias  → ``HaloGrids``
    3. ``cl_yy_1h_2h()``      — integration → (cl_1h, cl_2h)

    All stages are pure JAX, so the full chain is compatible with
    ``jax.grad``, ``jax.jacfwd``, ``jax.jacrev``, and ``jax.jit``.

    Parameters
    ----------
    ell : 1-d array
        Multipoles.
    cosmo : CosmoParams
        Cosmological parameters.
    profile_params : ProfileParamsA10 or ProfileParamsB12, optional
        Profile parameters.  ``None`` uses defaults for the chosen profile.
    profile : str
        ``'arnaud10'`` (default) or ``'battaglia12'``.
    z_grid : 1-d array, optional
        Redshift grid (default: ``geomspace(5e-3, 5, n_z)``).
    n_z : int
        Number of z points when *z_grid* is not given.
    cosmo_model : str
        Cosmological model label for the emulators.
    delta_crit : float
        Spherical-overdensity definition w.r.t. critical density
        (500 for arnaud10, 200 for battaglia12).
    m_min, m_max : float
        Mass range in M_sun.
    n_m : int
        Number of mass bins.

    Returns
    -------
    cl_1h, cl_2h : 1-d arrays, shape (n_ell,)
        Dimensionless C_ell (1-halo and 2-halo terms).
    """
    # -- Build params dict for the existing pipeline ---------------------
    params = {
        'omega_b':        cosmo.omega_b,
        'omega_cdm':      cosmo.omega_cdm,
        'H0':             cosmo.H0,
        'tau_reio':        cosmo.tau_reio,
        'ln10^{10}A_s':   cosmo.ln10_10_As,
        'n_s':            cosmo.n_s,
        'm_ncdm':         cosmo.m_ncdm,
    }

    # -- Stage 1: emulators → CosmoGrids --------------------------------
    cg = build_cosmo_grids(params, z_grid=z_grid, n_z=n_z,
                           cosmo_model=cosmo_model)

    # -- Stage 2: HMF/bias → HaloGrids ----------------------------------
    hg = build_halo_grids(cg, params, delta_crit=delta_crit,
                          m_min=m_min, m_max=m_max, n_m=n_m)

    # -- Stage 3: integration → C_ell -----------------------------------
    pp_dict = None
    if profile_params is not None:
        pp_dict = profile_params._asdict()

    cl_1h, cl_2h = cl_yy_1h_2h(ell, cg, hg, params,
                                profile=profile,
                                profile_params=pp_dict)

    return cl_1h, cl_2h
