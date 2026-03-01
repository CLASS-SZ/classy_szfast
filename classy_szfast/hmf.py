"""
Halo mass function and linear bias (JAX).

Builds a ``HaloGrids`` container from ``CosmoGrids`` — one emulator
call, then all downstream integrals are pure JAX.

Unit system — *physical (h-free)* throughout
=============================================

========== ============================== ==============
Quantity   Symbol                          Unit
========== ============================== ==============
M          halo mass                       M_sun
R_L        Lagrangian radius               Mpc
dn/dlnM    halo mass function              1 / Mpc³
b          linear halo bias                dimensionless
rho_m      mean matter density (z = 0)     M_sun / Mpc³
rho_crit   critical density (z)            M_sun / Mpc³
========== ============================== ==============

Design for JAX performance
==========================
* ``build_halo_grids(cosmo_grids, params)`` interpolates σ(R) → σ(M)
  once and returns a ``HaloGrids`` named-tuple.
* Every field of ``HaloGrids`` is a plain ``jnp.array``.
* ``HaloGrids`` is a JAX pytree (NamedTuple), so it can be passed
  directly into ``@jax.jit`` / ``jax.grad`` / ``jax.vmap`` functions.

Tinker 2008 multiplicity function
----------------------------------
The HMF is

    dn/dlnM = (ρ̄_m / M) × f(σ) × |dlnσ⁻¹/dlnM|

where ``f(σ)`` is the Tinker 2008 fitting function (Table 2 + redshift
evolution).  The ``tinker08_hmf`` function returns ``f(σ) / 2`` (the
factor of ½ compensates for using ``dlnσ²/dlnR`` instead of
``2 × dlnσ⁻¹/dlnR`` in the chain rule), so the HMF formula becomes

    dn/dlnM = (ρ̄_m / M) × hmf_f × (1/3) × |dlnσ²/dlnR|

with ``dlnM = 3 dlnR``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cosmology import CosmoGrids

jax.config.update("jax_enable_x64", True)


# ===================================================================
# Physical constants
# ===================================================================

DELTA_C = 1.686470199841145      # linear collapse threshold (3/20)(12π)^{2/3}

_G_SI    = 6.67428e-11           # m³ kg⁻¹ s⁻²  (matching CLASS value)
_c_SI    = 2.99792458e8          # m s⁻¹
_Msun_kg = 1.98855e30            # kg  (matching CLASS value)
_Mpc_m   = 3.085677581282e22     # m / Mpc


# ===================================================================
# Public data container
# ===================================================================

class HaloGrids(NamedTuple):
    """Precomputed HMF, bias, and support quantities on a (z, M) grid.

    All masses are in M_sun (physical, not M_sun/h).

    Axes convention
    ---------------
    ============== ================== ==============
    Field          Shape              Unit
    ============== ================== ==============
    lnM            (n_m,)             ln(M_sun)
    rho_m0         scalar             M_sun / Mpc³
    rho_crit_z     (n_z,)             M_sun / Mpc³
    Omega_m_z      (n_z,)             —
    sigma_m        (n_z, n_m)         —
    nu             (n_z, n_m)         δ_c / σ
    dndlnm         (n_z, n_m)         1 / Mpc³
    bias           (n_z, n_m)         —
    ============== ================== ==============
    """
    lnM:        jax.Array   # (n_m,)
    rho_m0:     jax.Array   # scalar
    rho_crit_z: jax.Array   # (n_z,)
    Omega_m_z:  jax.Array   # (n_z,)
    sigma_m:    jax.Array   # (n_z, n_m)
    nu:         jax.Array   # (n_z, n_m)
    dndlnm:     jax.Array   # (n_z, n_m)
    bias:       jax.Array   # (n_z, n_m)


# ===================================================================
# Tinker 2008 multiplicity function
# ===================================================================

_T08_DELTA_TAB = jnp.log10(jnp.array([200., 300., 400., 600., 800.,
                                        1200., 1600., 2400., 3200.]))
_T08_A_TAB = jnp.array([0.186, 0.200, 0.212, 0.218, 0.248,
                          0.255, 0.260, 0.260, 0.260])
_T08_a_TAB = jnp.array([1.47, 1.52, 1.56, 1.61, 1.87,
                          2.13, 2.30, 2.53, 2.66])
_T08_b_TAB = jnp.array([2.57, 2.25, 2.05, 1.87, 1.59,
                          1.51, 1.46, 1.44, 1.41])
_T08_c_TAB = jnp.array([1.19, 1.27, 1.34, 1.45, 1.58,
                          1.80, 1.97, 2.24, 2.44])


def tinker08_hmf(sigma, z, delta_mean):
    """Tinker 2008 multiplicity function (including ½ prefactor).

    Parameters
    ----------
    sigma      : (n_z, n_m)  rms fluctuation
    z          : (n_z,)      redshift
    delta_mean : (n_z,)      overdensity w.r.t. mean matter density

    Returns
    -------
    f : (n_z, n_m) — f(σ)/2 (see module docstring for the convention)
    """
    log_delta = jnp.log10(delta_mean)

    Ap = (jnp.interp(log_delta, _T08_DELTA_TAB, _T08_A_TAB)
          * (1.0 + z) ** (-0.14))
    a = (jnp.interp(log_delta, _T08_DELTA_TAB, _T08_a_TAB)
         * (1.0 + z) ** (-0.06))
    b = (jnp.interp(log_delta, _T08_DELTA_TAB, _T08_b_TAB)
         * (1.0 + z) ** (-jnp.power(
             10.0, -jnp.power(0.75 / jnp.log10(delta_mean / 75.0), 1.2))))
    c = jnp.interp(log_delta, _T08_DELTA_TAB, _T08_c_TAB)

    return (0.5 * Ap[:, None]
            * (jnp.power(sigma / b[:, None], -a[:, None]) + 1.0)
            * jnp.exp(-c[:, None] / sigma ** 2))


# ===================================================================
# Tinker 2010 linear halo bias
# ===================================================================

def tinker10_bias(nu, delta):
    """Tinker et al. 2010 first-order halo bias b(ν).

    Parameters
    ----------
    nu    : array, peak height δ_c / σ
    delta : float or array, overdensity w.r.t. mean matter density

    Returns
    -------
    b : same shape as nu
    """
    y = jnp.log10(delta)
    A = 1.0 + 0.24 * y * jnp.exp(-(4.0 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * jnp.exp(-(4.0 / y) ** 4)
    c = 2.4
    dc = DELTA_C
    return 1.0 - A * nu ** a / (nu ** a + dc ** a) + B * nu ** b + C * nu ** c


# ===================================================================
# Builder
# ===================================================================

def build_halo_grids(cg: CosmoGrids,
                     params: dict,
                     delta_crit: float = 200.0,
                     m_min: float = 1e10,
                     m_max: float = 5e16,
                     n_m: int = 200) -> HaloGrids:
    """Build HMF and bias grids from a ``CosmoGrids`` container.

    Parameters
    ----------
    cg : CosmoGrids
        Output of ``cosmology.build()``.
    params : dict
        Cosmological parameters (must include ``omega_b``, ``omega_cdm``,
        ``H0``).
    delta_crit : float
        Spherical-overdensity definition w.r.t. **critical** density
        (200 for M_200c, 500 for M_500c).  Converted to mean-density
        overdensity internally: Δ_mean(z) = Δ_crit / Ω_m(z).
    m_min, m_max : float
        Mass range in **M_sun** (physical).
    n_m : int
        Number of mass bins (uniform in ln M).

    Returns
    -------
    HaloGrids
        Named-tuple of JAX arrays, ready for downstream JIT functions.
    """
    h = params['H0'] / 100.0
    Omega_b   = params['omega_b'] / h ** 2
    Omega_cdm = params['omega_cdm'] / h ** 2
    Omega_ncdm = params.get('m_ncdm', 0.06) / (93.14 * h ** 2)
    Omega_m = Omega_b + Omega_cdm + Omega_ncdm

    # -- Critical density at z = 0  [M_sun / Mpc³] -----------------------
    H0_si = params['H0'] * 1e3 / _Mpc_m          # H₀ in 1/s
    rho_crit_0 = (3.0 * H0_si ** 2
                  / (8.0 * jnp.pi * _G_SI)
                  * _Mpc_m ** 3 / _Msun_kg)       # M_sun / Mpc³
    rho_m0 = Omega_m * rho_crit_0                  # M_sun / Mpc³

    # -- Critical density at z  [M_sun / Mpc³] ----------------------------
    #    H(z) in 1/s:  Hz [1/Mpc] × c [m/s] / Mpc [m]  (see cosmology.py)
    Hz_si = cg.Hz * _c_SI / _Mpc_m                # (n_z,) [1/s]
    rho_crit_z = (3.0 * Hz_si ** 2
                  / (8.0 * jnp.pi * _G_SI)
                  * _Mpc_m ** 3 / _Msun_kg)        # (n_z,) M_sun / Mpc³

    # -- Omega_m(z) -------------------------------------------------------
    Omega_m_z = rho_m0 * (1.0 + cg.z) ** 3 / rho_crit_z   # (n_z,)

    # -- Mass grid in M_sun (uniform in ln M) -----------------------------
    lnM = jnp.linspace(jnp.log(m_min), jnp.log(m_max), n_m)
    M = jnp.exp(lnM)                               # (n_m,) M_sun

    # -- Lagrangian radius:  M = (4π/3) ρ̄_m,0 R³ -------------------------
    R_lag = jnp.power(3.0 * M / (4.0 * jnp.pi * rho_m0),
                      1.0 / 3.0)                    # (n_m,) Mpc

    # -- Interpolate σ(R, z) from CosmoGrids → σ(M, z) -------------------
    log_R_cg = jnp.log(cg.R)                       # (n_R,)
    log_R_m  = jnp.log(R_lag)                       # (n_m,)
    log_sigma_cg = jnp.log(cg.sigma)                # (n_z, n_R)

    sigma_m = jnp.exp(jax.vmap(
        lambda lnsig_z: jnp.interp(log_R_m, log_R_cg, lnsig_z)
    )(log_sigma_cg))                                 # (n_z, n_m)

    # -- dlnσ²/dlnR via finite differences on the mass grid ---------------
    #    lnM is uniform ⟹ lnR is uniform (R ∝ M^{1/3})
    lnsig2 = 2.0 * jnp.log(sigma_m)                 # (n_z, n_m)
    dlnR = (1.0 / 3.0) * (lnM[1] - lnM[0])

    # Central differences (interior)
    dlnsig2_center = (lnsig2[:, 2:] - lnsig2[:, :-2]) / (2.0 * dlnR)
    # Forward / backward differences (edges)
    dlnsig2_left  = (lnsig2[:, 1:2] - lnsig2[:, 0:1]) / dlnR
    dlnsig2_right = (lnsig2[:, -1:] - lnsig2[:, -2:-1]) / dlnR

    dlnsig2_dlnR = jnp.concatenate(
        [dlnsig2_left, dlnsig2_center, dlnsig2_right], axis=1)  # (n_z, n_m)

    # -- Tinker 2008 HMF --------------------------------------------------
    delta_mean = delta_crit / Omega_m_z              # (n_z,)

    hmf_f = tinker08_hmf(sigma_m, cg.z, delta_mean)  # (n_z, n_m), includes ½

    # dn/dlnM = (ρ̄_m / M) × hmf_f × (1/3) × |dlnσ²/dlnR|
    #         = (ρ̄_m / M) × [f(σ)/2] × (1/3) × |dlnσ²/dlnR|
    #         = (ρ̄_m / M) × f(σ) × (1/6) × |dlnσ²/dlnR|
    #         = (ρ̄_m / M) × f(σ) × |dlnσ⁻¹/dlnM|             ✓
    dndlnm = ((rho_m0 / M[None, :])
              * hmf_f
              * (1.0 / 3.0)
              * jnp.abs(dlnsig2_dlnR))               # (n_z, n_m) 1/Mpc³

    # -- Peak height and bias ---------------------------------------------
    nu = DELTA_C / sigma_m                            # (n_z, n_m)
    bias = tinker10_bias(nu, delta_mean[:, None])     # (n_z, n_m)

    return HaloGrids(
        lnM=lnM,
        rho_m0=jnp.float64(rho_m0),
        rho_crit_z=rho_crit_z,
        Omega_m_z=Omega_m_z,
        sigma_m=sigma_m,
        nu=nu,
        dndlnm=dndlnm,
        bias=bias,
    )
