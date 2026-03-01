"""
Angular power spectrum C_ell^yy (JAX).

Takes ``CosmoGrids`` + ``HaloGrids`` + multipole array and returns
the tSZ angular power spectrum C_ell^yy (1-halo + 2-halo).

Profile Fourier transforms are precomputed at import time as lookup
tables (1-D for Arnaud 2010, 2-D for Battaglia 2012).

Unit system — *physical (h-free)* throughout
=============================================

The y_ell formula (see Bolliet et al. 2018, Eq. 5):

    y_ell(M, z, ell) = prefac × P × ũ(k_ell) × (1+z)² / χ²

where, for Battaglia 2012 (thermal pressure):

    prefac = PREFAC_Y  = (σ_T / m_e c²) × PE_FACTOR × MPC_TO_CM
    P      = P_Δ       = 3 G M² f_b / (8π r_Δ⁴)               [eV/cm³]

and for Arnaud 2010 (electron pressure directly):

    prefac = PREFAC_Y_E = (σ_T / m_e c²) × MPC_TO_CM   (no PE_FACTOR)
    P      = P_500       from Arnaud et al. 2010 Eq. 12  [eV/cm³]

Common:

    k_ell    = (ell + 0.5) × (1 + z) / χ          [1/Mpc]
    ũ        = 4π r³ g(k r_s)                       [Mpc³]

Design for JAX performance
==========================
* All lookup-table interpolations use ``jnp.searchsorted`` — no Python
  loops, fully JIT-compatible.
* ``cl_yy_1h_2h`` is a pure function of ``(ell, CosmoGrids, HaloGrids,
  params)`` — no side effects, no mutable state.

Pressure profile models
-----------------------
* ``'battaglia12'`` — Battaglia et al. 2012 gNFW (mass- and z-dependent
  P0, xc, beta).  Uses 2-D FT table ``g(s, beta)``.
* ``'arnaud10'`` — Arnaud et al. 2010 universal profile (fixed gNFW
  parameters).  Uses 1-D FT table ``g(s)``.
"""

from __future__ import annotations

import warnings as _warnings

import jax
import jax.numpy as jnp
from mcfit import SphericalBessel

from .cosmology import CosmoGrids
from .hmf import HaloGrids, _G_SI, _Msun_kg, _Mpc_m, _c_SI

jax.config.update("jax_enable_x64", True)


# ===================================================================
# Physical constants
# ===================================================================

SIGMA_T    = 6.6524587321e-29    # m²  (Thomson cross section)
M_E_C2     = 0.510998946e6       # eV  (electron rest energy)
MPC_TO_CM  = 3.085677581282e24   # cm / Mpc
_eV_J      = 1.602176487e-19     # J / eV

XH         = 0.76                # hydrogen mass fraction
PE_FACTOR  = (2.0 * XH + 2.0) / (5.0 * XH + 3.0)   # P_e / P_th ≈ 0.5176

_SIGMA_T_CGS = SIGMA_T * 1e4    # cm²

# Master prefactor  [cm³ / (eV × Mpc)]
# When multiplied by P_Δ [eV/cm³] × ũ [Mpc³] / χ² [Mpc²]  → dimensionless
PREFAC_Y = _SIGMA_T_CGS / M_E_C2 * PE_FACTOR * MPC_TO_CM

# For profiles that give electron pressure directly (e.g. Arnaud 2010):
# no PE_FACTOR needed since P_e is given, not P_thermal.
PREFAC_Y_E = _SIGMA_T_CGS / M_E_C2 * MPC_TO_CM


# ===================================================================
# Battaglia 2012 gNFW profile parameters
# ===================================================================

_B12_DEFAULTS = {
    'P0_A': 18.1,   'P0_am': 0.154,    'P0_az': -0.758,
    'xc_A': 0.497,  'xc_am': -0.00865, 'xc_az': 0.731,
    'beta_A': 4.35,  'beta_am': 0.0393,  'beta_az': 0.415,
}


def _b12_params(M200c, z, profile_params=None):
    """Battaglia 2012 best-fit gNFW parameters.

    Parameters: M200c in M_sun, z scalar or array.
    profile_params: dict overriding fitting coefficients, or None.
    Returns (P0, xc, beta) broadcast-compatible.
    """
    pp = profile_params or {}
    m14 = M200c / 1e14
    P0   = (pp.get('P0_A', _B12_DEFAULTS['P0_A'])
            * m14 ** pp.get('P0_am', _B12_DEFAULTS['P0_am'])
            * (1.0 + z) ** pp.get('P0_az', _B12_DEFAULTS['P0_az']))
    xc   = (pp.get('xc_A', _B12_DEFAULTS['xc_A'])
            * m14 ** pp.get('xc_am', _B12_DEFAULTS['xc_am'])
            * (1.0 + z) ** pp.get('xc_az', _B12_DEFAULTS['xc_az']))
    beta = (pp.get('beta_A', _B12_DEFAULTS['beta_A'])
            * m14 ** pp.get('beta_am', _B12_DEFAULTS['beta_am'])
            * (1.0 + z) ** pp.get('beta_az', _B12_DEFAULTS['beta_az']))
    return P0, xc, beta


# ===================================================================
# Arnaud 2010 universal profile constants
# ===================================================================

_A10_P0    = 8.130
_A10_C500  = 1.156
_A10_GAMMA = 0.3292
_A10_ALPHA = 1.0620
_A10_BETA  = 5.4807


# ===================================================================
# Precomputed profile FT lookup tables  (built once at import time)
# ===================================================================

_TABLE_N_U = 256
_TABLE_U_GRID = jnp.geomspace(1e-5, 100.0, _TABLE_N_U)

_TABLE_N_BETA = 100
_TABLE_BETAS  = jnp.linspace(3.0, 13.0, _TABLE_N_BETA)

with _warnings.catch_warnings():
    _warnings.filterwarnings("ignore",
                             message="use backend='jax' if desired")
    _TABLE_SBT = SphericalBessel(_TABLE_U_GRID, nu=0, lowring=True,
                                  backend='jax')

_TABLE_S_GRID = jnp.array(_TABLE_SBT.y)        # output s-grid
_LOG_TABLE_S  = jnp.log(_TABLE_S_GRID)


def _build_b12_ft_table():
    """Precompute g(s, beta) for Battaglia 2012 on a 2-D grid."""
    profiles = (_TABLE_U_GRID[None, :] ** (-0.3)
                * (1.0 + _TABLE_U_GRID[None, :]) ** (-_TABLE_BETAS[:, None]))
    _, g = _TABLE_SBT(profiles, extrap=True)
    return g * jnp.sqrt(jnp.pi / 2.0)             # (n_beta, n_s)


def _build_a10_ft_table():
    """Precompute g(s) for Arnaud 2010 on a 1-D grid."""
    kernel = (_TABLE_U_GRID ** (-_A10_GAMMA)
              * (1.0 + _TABLE_U_GRID ** _A10_ALPHA)
              ** ((_A10_GAMMA - _A10_BETA) / _A10_ALPHA))
    _, g = _TABLE_SBT(kernel, extrap=True)
    return g * jnp.sqrt(jnp.pi / 2.0)             # (n_s,)


_B12_G_TABLE = _build_b12_ft_table()
_A10_G_TABLE = _build_a10_ft_table()


# ===================================================================
# Internal helpers
# ===================================================================

def _P_delta(M, r_delta, f_b):
    """Thermal pressure P_Δ in eV/cm³.

    General formula: P = 3 G M² f_b / (8π r⁴)
    Works for any overdensity (200c, 500c, …).

    Parameters: M in M_sun, r_delta in Mpc.
    """
    M_kg = M * _Msun_kg
    r_m  = r_delta * _Mpc_m
    P_SI = 3.0 * _G_SI * M_kg ** 2 * f_b / (8.0 * jnp.pi * r_m ** 4)
    return P_SI / _eV_J * 1e-6   # Pa → eV/cm³


def _P500_arnaud10(M_h, Ez, h, alpha_p=0.12):
    """Arnaud 2010 electron pressure P_500 in eV/cm³.

    Eq. 12 of Arnaud et al. 2010, converted from keV to eV.
    This is the *electron* pressure — no PE_FACTOR needed.
    P0 is NOT included here (it enters through the profile FT).

    Parameters
    ----------
    M_h : array — M_500c in M_sun/h
    Ez  : array — E(z) = H(z)/H0
    h   : float — H0/100
    alpha_p : float — mass-slope correction (default 0.12)
    """
    # 1.65e-3 keV/cm³ = 1.65 eV/cm³
    return (1.65 * (h / 0.7) ** 2 * Ez ** (8.0 / 3.0)
            * (M_h / (3.0e14 * 0.7)) ** (2.0 / 3.0 + alpha_p)
            * (0.7 / h) ** 1.5)


def _interp_1d_log(log_x, log_x_table, y_table):
    """Linear interpolation in log-x space (1-D), vectorised."""
    n = log_x_table.shape[0]
    ix = jnp.searchsorted(log_x_table, log_x) - 1
    ix = jnp.clip(ix, 0, n - 2)
    t = ((log_x - log_x_table[ix])
         / (log_x_table[ix + 1] - log_x_table[ix]))
    return y_table[ix] * (1.0 - t) + y_table[ix + 1] * t


# ===================================================================
# cl_yy — public API
# ===================================================================

def cl_yy_1h_2h(ell: jax.Array,
                cg: CosmoGrids,
                hg: HaloGrids,
                params: dict,
                profile: str = 'battaglia12',
                profile_params: dict | None = None) -> tuple[jax.Array, jax.Array]:
    """Compute C_ell^yy (1-halo and 2-halo terms).

    Parameters
    ----------
    ell : 1-d array
        Multipoles.
    cg : CosmoGrids
        Output of ``cosmology.build()``.
    hg : HaloGrids
        Output of ``hmf.build_halo_grids()``.
    params : dict
        Cosmological parameters (needs ``omega_b``, ``omega_cdm``, ``H0``).
    profile : str
        ``'battaglia12'`` (default) or ``'arnaud10'``.
    profile_params : dict, optional
        Override Arnaud 2010 profile parameters.  Recognised keys:
        ``P0``, ``c500``, ``gamma``, ``alpha``, ``beta``.
        When ``gamma``, ``alpha``, or ``beta`` differ from their
        defaults the FT lookup table is recomputed inline (~0.5 ms).

    Returns
    -------
    cl_1h, cl_2h : 1-d arrays, shape (n_ell,)
        Dimensionless C_ell.
    """
    h = params['H0'] / 100.0
    Omega_b   = params['omega_b'] / h ** 2
    Omega_cdm = params['omega_cdm'] / h ** 2
    Omega_ncdm = params.get('m_ncdm', 0.06) / (93.14 * h ** 2)
    Omega_m = Omega_b + Omega_cdm + Omega_ncdm
    f_b = Omega_b / Omega_m

    n_ell = ell.shape[0]
    n_z = cg.z.shape[0]
    n_m = hg.lnM.shape[0]

    M = jnp.exp(hg.lnM)                              # (n_m,) M_sun

    # ── y_ell computation ────────────────────────────────────────────────
    if profile == 'arnaud10':
        # Extract profile parameters (fall back to module defaults)
        P0    = profile_params.get('P0', _A10_P0) if profile_params else _A10_P0
        c500  = profile_params.get('c500', _A10_C500) if profile_params else _A10_C500
        gamma = profile_params.get('gamma', _A10_GAMMA) if profile_params else _A10_GAMMA
        alpha = profile_params.get('alpha', _A10_ALPHA) if profile_params else _A10_ALPHA
        beta  = profile_params.get('beta', _A10_BETA) if profile_params else _A10_BETA

        # Recompute FT table when shape params differ from defaults
        shape_params_vary = (profile_params is not None
                             and any(k in profile_params
                                     for k in ('gamma', 'alpha', 'beta')))
        if shape_params_vary:
            kernel = (_TABLE_U_GRID ** (-gamma)
                      * (1.0 + _TABLE_U_GRID ** alpha)
                      ** ((gamma - beta) / alpha))
            _, g_table = _TABLE_SBT(kernel, extrap=True)
            g_table = g_table * jnp.sqrt(jnp.pi / 2.0)
        else:
            g_table = _A10_G_TABLE

        # r_500c(M, z) [Mpc]
        r_delta = jnp.power(
            3.0 * M[None, :] / (4.0 * jnp.pi * 500.0
                                 * hg.rho_crit_z[:, None]),
            1.0 / 3.0)                                # (n_z, n_m)

        # s = k_phys × r_500c / c500,  k_phys = (ell+0.5)(1+z)/chi = (ell+0.5)/d_A
        s_query = ((ell[:, None, None] + 0.5)
                   * r_delta[None, :, :]
                   * (1.0 + cg.z)[None, :, None]
                   / (c500 * cg.chi[None, :, None]))

        # 1-D interpolation
        log_sq = jnp.log(jnp.clip(s_query, 1e-30)).ravel()
        g_interp = _interp_1d_log(
            log_sq, _LOG_TABLE_S, g_table
        ).reshape(n_ell, n_z, n_m)

        # ũ = 4π P0 (r/c500)³ g(s)  [Mpc³]
        r_over_c = r_delta / c500
        u_at_ell = (4.0 * jnp.pi * P0
                    * r_over_c[None, :, :] ** 3
                    * g_interp)

        # Arnaud 2010 electron pressure P_500 [eV/cm³]
        H0_over_c = h * 100.0 * 1e3 / _c_SI   # H0/c in 1/Mpc
        Ez = cg.Hz / H0_over_c                  # (n_z,)
        M_h = M * h                              # M_sun/h
        P_delta_grid = _P500_arnaud10(M_h[None, :], Ez[:, None], h)

    else:  # battaglia12
        # r_200c(M, z) [Mpc]
        r_delta = jnp.power(
            3.0 * M[None, :] / (4.0 * jnp.pi * 200.0
                                 * hg.rho_crit_z[:, None]),
            1.0 / 3.0)

        P0, xc, beta_vals = _b12_params(M[None, :], cg.z[:, None],
                                        profile_params=profile_params)

        # s = k_phys × r_200c × xc,  k_phys = (ell+0.5)(1+z)/chi
        s_query = ((ell[:, None, None] + 0.5)
                   * r_delta[None, :, :] * xc[None, :, :]
                   * (1.0 + cg.z)[None, :, None]
                   / cg.chi[None, :, None])

        # 2-D bilinear interpolation in (log_s, beta)
        n_s = _TABLE_S_GRID.shape[0]
        n_beta = _TABLE_BETAS.shape[0]

        beta_flat = beta_vals.ravel()
        ib = jnp.clip(jnp.searchsorted(_TABLE_BETAS, beta_flat) - 1,
                       0, n_beta - 2)
        tb = ((beta_flat - _TABLE_BETAS[ib])
              / (_TABLE_BETAS[ib + 1] - _TABLE_BETAS[ib]))
        ib_b = jnp.tile(ib, n_ell)
        tb_b = jnp.tile(tb, n_ell)

        log_sq = jnp.log(jnp.clip(s_query, 1e-30)).ravel()
        is_ = jnp.clip(jnp.searchsorted(_LOG_TABLE_S, log_sq) - 1,
                        0, n_s - 2)
        ts = ((log_sq - _LOG_TABLE_S[is_])
              / (_LOG_TABLE_S[is_ + 1] - _LOG_TABLE_S[is_]))

        g00 = _B12_G_TABLE[ib_b, is_]
        g01 = _B12_G_TABLE[ib_b, is_ + 1]
        g10 = _B12_G_TABLE[ib_b + 1, is_]
        g11 = _B12_G_TABLE[ib_b + 1, is_ + 1]
        g_interp = ((1 - tb_b) * (1 - ts) * g00
                    + (1 - tb_b) * ts * g01
                    + tb_b * (1 - ts) * g10
                    + tb_b * ts * g11).reshape(n_ell, n_z, n_m)

        # ũ = 4π r³ P0 xc³ g(s, β)  [Mpc³]
        u_at_ell = (4.0 * jnp.pi * r_delta[None, :, :] ** 3
                    * P0[None, :, :] * xc[None, :, :] ** 3
                    * g_interp)

        # P_200c [eV/cm³]
        P_delta_grid = _P_delta(M[None, :], r_delta, f_b)

    # ── y_ell = prefac × P × ũ × (1+z)² / χ² ────────────────────────
    # A10: P_500 is electron pressure → use PREFAC_Y_E (no PE_FACTOR)
    # B12: P_200c is thermal pressure → use PREFAC_Y   (with PE_FACTOR)
    prefac = PREFAC_Y_E if profile == 'arnaud10' else PREFAC_Y
    onepz2_chi2 = (1.0 + cg.z) ** 2 / cg.chi ** 2    # (n_z,)
    y_ell = (prefac * P_delta_grid[None, :, :]
             * u_at_ell
             * onepz2_chi2[None, :, None])             # (n_ell, n_z, n_m)

    # ── Volume element dV/dz/dΩ = χ²/H ─────────────────────────────────
    dVdzdOmega = cg.chi ** 2 / cg.Hz                   # (n_z,)

    # ── 1-halo: C_ell^1h = ∫dz (dV/dΩ) ∫dlnM (dn/dlnM) y²  ───────────
    integ_1h = hg.dndlnm[None, :, :] * y_ell ** 2      # (n_ell, n_z, n_m)
    I_m_1h = jnp.trapezoid(integ_1h, hg.lnM, axis=2)  # (n_ell, n_z)
    cl_1h = jnp.trapezoid(dVdzdOmega[None, :] * I_m_1h,
                           cg.z, axis=1)                # (n_ell,)

    # ── 2-halo: C_ell^2h = ∫dz (dV/dΩ) P_lin [∫dlnM (dn/dlnM) b y]² ──
    integ_2h = (hg.dndlnm[None, :, :] * hg.bias[None, :, :]
                * y_ell)
    I_m_2h = jnp.trapezoid(integ_2h, hg.lnM, axis=2)  # (n_ell, n_z)

    # P_lin(k_ell, z) via vectorised interpolation
    log_k_pk = jnp.log(cg.k)                           # (n_k,)
    log_pk = jnp.log(cg.pk)                             # (n_z, n_k)

    # Comoving Limber wavenumber (NOT physical — no (1+z) factor here)
    k_ell_z = (ell[:, None] + 0.5) / cg.chi[None, :]    # (n_ell, n_z) 1/Mpc
    log_kq = jnp.log(k_ell_z).ravel()

    ik = jnp.clip(jnp.searchsorted(log_k_pk, log_kq) - 1,
                   0, cg.k.shape[0] - 2)
    tk = ((log_kq - log_k_pk[ik])
          / (log_k_pk[ik + 1] - log_k_pk[ik]))
    z_flat = jnp.tile(jnp.arange(n_z), n_ell)
    pk_lo = log_pk[z_flat, ik]
    pk_hi = log_pk[z_flat, ik + 1]
    pk_at_ell = jnp.exp(pk_lo + tk * (pk_hi - pk_lo)
                        ).reshape(n_ell, n_z)

    cl_2h = jnp.trapezoid(
        dVdzdOmega[None, :] * pk_at_ell * I_m_2h ** 2,
        cg.z, axis=1)                                   # (n_ell,)

    return cl_1h, cl_2h
