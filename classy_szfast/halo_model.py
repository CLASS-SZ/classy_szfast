"""
Pure-JAX halo model for the tSZ angular power spectrum C_ell^yy.

All functions are stateless (arrays in, arrays out) and compatible with
jax.jit, jax.grad, jax.vmap.

Unit conventions (matching classy_szfast):
  - k in 1/Mpc, P(k) in Mpc^3, masses in M_sun, distances in Mpc
  - H(z) in 1/Mpc (= H_phys / c), rho_crit in M_sun/Mpc^3
  - P_200c in eV/cm^3, C_ell^yy dimensionless

The y_ell formula (Bolliet et al. 2018, Eq. 5):
  y_ell(M,z) = (sigma_T / m_e c^2) * (4 pi r_Delta / ell_Delta^2)
               * int dx x^2 P_e(x) j_0(ell x / ell_Delta)
  with ell_Delta = d_A / r_Delta, P_e = PE_FACTOR * P_Delta * f_profile(x).

  Equivalently:
  y_ell = PREFAC_Y * P_Delta * u_tilde(k_ell) * (1+z)^2 / chi^2
  where PREFAC_Y = sigma_T_cgs / (m_e c^2) * PE_FACTOR * MPC_TO_CM
  and u_tilde = 4 pi r_Delta^3 * f_tilde(k_ell * r_Delta)  [Mpc^3].
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from mcfit import SphericalBessel
from .utils import Const

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SIGMA_T = 6.6524587321e-29       # m^2  (Thomson cross section)
M_E_C2 = 0.510998946e6          # eV   (electron rest mass energy)
MPC_TO_CM = 3.085677581282e24    # cm / Mpc
MPC_TO_M = 3.085677581282e22    # m  / Mpc
DELTA_C = 1.686470199841145      # (3/20)(12 pi)^{2/3}
XH = 0.76                        # hydrogen mass fraction
PE_FACTOR = (2.0 * XH + 2.0) / (5.0 * XH + 3.0)  # P_e / P_th ~ 0.5176

_G_SI = Const._G_               # m^3 kg^-1 s^-2
_c_SI = Const._c_               # m/s
_Msun_kg = Const._M_sun_        # kg

# Master prefactor for y_ell (constant):
# sigma_T [cm^2] / (m_e c^2 [eV]) * PE_FACTOR * MPC_TO_CM [cm/Mpc]
# Units: cm^2/eV * cm/Mpc = cm^3 / (eV * Mpc)
# When multiplied by P_Delta [eV/cm^3] * u_tilde [Mpc^3] / d_A^2 [Mpc^2]:
# => cm^3/(eV*Mpc) * eV/cm^3 * Mpc^3/Mpc^2 = dimensionless
_SIGMA_T_CGS = SIGMA_T * 1e4    # m^2 -> cm^2
PREFAC_Y = _SIGMA_T_CGS / M_E_C2 * PE_FACTOR * MPC_TO_CM


# ---------------------------------------------------------------------------
# 1. Battaglia 2012 gNFW pressure profile
# ---------------------------------------------------------------------------
def battaglia12_profile_params(M200c, z):
    """Battaglia 2012 best-fit parameters for the gNFW pressure profile.

    Parameters
    ----------
    M200c : array  (M_sun, *not* M_sun/h)
    z     : array  (redshift)

    Returns
    -------
    P0, xc, beta : arrays of same shape as broadcast(M200c, z)
    """
    m14 = M200c / 1e14
    P0   = 18.1  * m14**0.154    * (1.0 + z)**(-0.758)
    xc   = 0.497 * m14**(-0.00865) * (1.0 + z)**0.731
    beta = 4.35  * m14**0.0393   * (1.0 + z)**0.415
    return P0, xc, beta


def battaglia12_profile_x(x, M200c, z):
    """Normalised gNFW pressure profile  P(x) / P_200c.

    Parameters
    ----------
    x     : array, r / r_200c
    M200c : scalar or array  (M_sun)
    z     : scalar or array

    Returns
    -------
    profile : same shape as broadcast(x, M200c, z)
    """
    P0, xc, beta = battaglia12_profile_params(M200c, z)
    gamma = -0.3
    alpha = 1.0
    t = x / xc
    return P0 * t**gamma * (1.0 + t**alpha)**(-beta)


# ---------------------------------------------------------------------------
# 1b. Arnaud et al. 2010 universal pressure profile
# ---------------------------------------------------------------------------
# Fixed gNFW parameters (Table 1, "universal" column):
_A10_P0    = 8.130
_A10_C500  = 1.156
_A10_GAMMA = 0.3292
_A10_ALPHA = 1.0620
_A10_BETA  = 5.4807


def arnaud10_profile_x(x):
    """Arnaud et al. 2010 universal pressure profile  P(x) / P_500c.

    All parameters are fixed (no mass/redshift dependence).

    Parameters
    ----------
    x : array, r / r_500c

    Returns
    -------
    profile : same shape as x
    """
    u = _A10_C500 * x
    return (_A10_P0 * u ** (-_A10_GAMMA)
            * (1.0 + u ** _A10_ALPHA) ** ((_A10_GAMMA - _A10_BETA) / _A10_ALPHA))


# ---------------------------------------------------------------------------
# 1c. NFW mass conversion  (M_200m → M_500c via Duffy 2008 + Bryan-Norman)
# ---------------------------------------------------------------------------
def _nfw_mu(x):
    """NFW enclosed mass function: f(x) = ln(1+x) - x/(1+x)."""
    return jnp.log(1.0 + x) - x / (1.0 + x)


def duffy08_cvir(M_h, z):
    """Duffy et al. 2008 virial concentration c_vir(M, z).

    Parameters
    ----------
    M_h : array — virial mass in M_sun/h
    z   : array — redshift

    Returns
    -------
    c_vir : array
    """
    return 7.85 * (M_h / 2.0e12) ** (-0.081) * (1.0 + z) ** (-0.71)


def _delta_vir_BN98(Omega_m_z):
    """Bryan & Norman 1998 virial overdensity Δ_c w.r.t. critical density."""
    x = Omega_m_z - 1.0
    return 18.0 * jnp.pi ** 2 + 82.0 * x - 39.0 * x ** 2


def _m200m_to_m500c(M_200m, z, rho_crit_z, Omega_m_z, h):
    """Convert M_200m → M_500c via NFW profile with Duffy 2008 c(M,z).

    Follows the CLASS-SZ two-step algorithm:
      1. M_200m → M_vir  (fixed-point iteration)
      2. M_vir  → M_500c (fixed-point iteration)

    Parameters
    ----------
    M_200m : array (n_z, n_m) — mass in M_sun
    z      : array (n_z,)
    rho_crit_z : array (n_z,) — critical density in M_sun/Mpc^3
    Omega_m_z  : array (n_z,) — Omega_m(z)
    h      : float

    Returns
    -------
    M_500c : array (n_z, n_m) — mass in M_sun
    """
    # Broadcast M_200m to (n_z, n_m) for consistent shapes
    M_200m = jnp.broadcast_to(M_200m, (z.shape[0], M_200m.shape[-1]))

    Delta_vir = _delta_vir_BN98(Omega_m_z)           # (n_z,)
    rho_200m = 200.0 * Omega_m_z * rho_crit_z        # (n_z,)

    # r_200m is fixed (known from M_200m)
    r_200m = jnp.power(
        3.0 * M_200m / (4.0 * jnp.pi * rho_200m[:, None]),
        1.0 / 3.0)                                   # (n_z, n_m)

    # --- Step 1: M_200m → M_vir (fixed-point) ---
    # M_200m = M_vir * f(r_200m / r_s) / f(c_vir)
    # ⟹ M_vir = M_200m * f(c_vir) / f(r_200m / r_s)
    def _step_vir(_, M_vir):
        c_vir = duffy08_cvir(M_vir * h, z[:, None])
        r_vir = jnp.power(
            3.0 * M_vir / (4.0 * jnp.pi
                            * Delta_vir[:, None] * rho_crit_z[:, None]),
            1.0 / 3.0)
        r_s = r_vir / c_vir
        return M_200m * _nfw_mu(c_vir) / _nfw_mu(r_200m / r_s)

    M_vir = jax.lax.fori_loop(0, 15, _step_vir, M_200m)

    # --- Step 2: M_vir → M_500c (fixed-point) ---
    c_vir = duffy08_cvir(M_vir * h, z[:, None])      # (n_z, n_m)
    r_vir = jnp.power(
        3.0 * M_vir / (4.0 * jnp.pi
                         * Delta_vir[:, None] * rho_crit_z[:, None]),
        1.0 / 3.0)
    r_s = r_vir / c_vir                               # (n_z, n_m)
    f_cvir = _nfw_mu(c_vir)
    rho_500 = 500.0 * rho_crit_z                      # (n_z,)

    # M_500c = M_vir * f(r_500c / r_s) / f(c_vir)
    def _step_500(_, M_cur):
        r_500c = jnp.power(
            3.0 * M_cur / (4.0 * jnp.pi * rho_500[:, None]), 1.0 / 3.0)
        return M_vir * _nfw_mu(r_500c / r_s) / f_cvir

    return jax.lax.fori_loop(0, 15, _step_500, 0.7 * M_vir)


# ---------------------------------------------------------------------------
# 2. Tinker 2010 first-order halo bias
# ---------------------------------------------------------------------------
def tinker10_bias(nu, delta=200.0):
    """First-order halo bias b(nu) from Tinker et al. 2010, Eq. 6.

    Parameters
    ----------
    nu    : array, peak height = delta_c / sigma  (NOT nu^2)
    delta : float, overdensity w.r.t. mean matter density

    Returns
    -------
    b : array, same shape as nu
    """
    y = jnp.log10(delta)
    A = 1.0 + 0.24 * y * jnp.exp(-(4.0 / y)**4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * jnp.exp(-(4.0 / y)**4)
    c = 2.4
    dc = DELTA_C
    return 1.0 - A * nu**a / (nu**a + dc**a) + B * nu**b + C * nu**c


# ---------------------------------------------------------------------------
# 3. Tinker 2008 mass function  (ported from notebook)
# ---------------------------------------------------------------------------
_T08_DELTA_TAB = jnp.log10(jnp.array([200., 300., 400., 600., 800.,
                                        1200., 1600., 2400., 3200.]))
_T08_A_TAB  = jnp.array([0.186, 0.200, 0.212, 0.218, 0.248,
                          0.255, 0.260, 0.260, 0.260])
_T08_a_TAB  = jnp.array([1.47, 1.52, 1.56, 1.61, 1.87,
                          2.13, 2.30, 2.53, 2.66])
_T08_b_TAB  = jnp.array([2.57, 2.25, 2.05, 1.87, 1.59,
                          1.51, 1.46, 1.44, 1.41])
_T08_c_TAB  = jnp.array([1.19, 1.27, 1.34, 1.45, 1.58,
                          1.80, 1.97, 2.24, 2.44])


def tinker08_hmf(sigma, z, delta_mean):
    """Tinker 2008 multiplicity function f(sigma).

    Parameters
    ----------
    sigma      : array, shape (n_z, n_m) — rms fluctuation
    z          : array, shape (n_z,)
    delta_mean : array, shape (n_z,) — overdensity w.r.t. mean

    Returns
    -------
    f : array, shape (n_z, n_m)
    """
    log_delta = jnp.log10(delta_mean)

    Ap = jnp.interp(log_delta, _T08_DELTA_TAB, _T08_A_TAB) * (1.0 + z)**(-0.14)
    a  = jnp.interp(log_delta, _T08_DELTA_TAB, _T08_a_TAB) * (1.0 + z)**(-0.06)
    b  = jnp.interp(log_delta, _T08_DELTA_TAB, _T08_b_TAB) * (1.0 + z)**(-jnp.power(
            10.0, -jnp.power(0.75 / jnp.log10(delta_mean / 75.0), 1.2)))
    c  = jnp.interp(log_delta, _T08_DELTA_TAB, _T08_c_TAB)

    return (0.5 * Ap[:, None]
            * (jnp.power(sigma / b[:, None], -a[:, None]) + 1.0)
            * jnp.exp(-c[:, None] / sigma**2))


# ---------------------------------------------------------------------------
# 4 & 5. Profile Fourier transform via mcfit SphericalBessel
# ---------------------------------------------------------------------------
_N_R_FT = 512
_R_GRID_FT = jnp.geomspace(1e-4, 6.0, _N_R_FT)  # x = r/r_200c


def profile_fourier_setup(r_grid=None):
    """Construct the mcfit SphericalBessel transform object.

    Parameters
    ----------
    r_grid : 1-d array, optional
        Radial grid in units of r_200c.  Defaults to a log-spaced grid
        covering [1e-4, 6].

    Returns
    -------
    sbt : mcfit.SphericalBessel instance (backend='jax')
    """
    if r_grid is None:
        r_grid = _R_GRID_FT
    return SphericalBessel(r_grid, nu=0, lowring=True, backend='jax')


# Build once at import time (uses only module-level constants).
import warnings as _warnings
with _warnings.catch_warnings():
    _warnings.filterwarnings("ignore", message="use backend='jax' if desired")
    _DEFAULT_SBT = profile_fourier_setup()


def profile_fourier(profile_on_r, r_200c, sbt=None):
    """Spherical Bessel FT of a radial pressure profile.

    Computes u_tilde(k) = 4pi * r_200c^3 * int dx x^2 f(x) j_0(kx) ,
    where the input profile f(x) = P(x)/P_200c is evaluated on the
    default x-grid.

    Parameters
    ----------
    profile_on_r : 1-d array, shape (n_r,)
        Normalised profile P(x)/P_200c on the x = r/r_200c grid.
    r_200c : scalar
        Physical virial radius in Mpc.
    sbt : SphericalBessel, optional

    Returns
    -------
    k : 1-d array (1/Mpc)
    u_tilde : 1-d array (Mpc^3, dimensionless profile units)
    """
    if sbt is None:
        sbt = _DEFAULT_SBT
    # SBT returns: q_values, G(q) = int f(x) j_0(qx) x^2 dx
    # mcfit uses unitary Hankel convention; correct by sqrt(pi/2)
    q, f_tilde = sbt(profile_on_r, extrap=True)
    f_tilde = f_tilde * jnp.sqrt(jnp.pi / 2.0)
    k = q / r_200c               # physical k in 1/Mpc
    u_tilde = 4.0 * jnp.pi * f_tilde * r_200c**3  # Mpc^3
    return k, u_tilde


# ---------------------------------------------------------------------------
# 4b. Precomputed profile FT lookup table
# ---------------------------------------------------------------------------
# The Battaglia profile FT factors as:
#   f_tilde(q) = P0 * xc^3 * g(q*xc, beta)
# where g(s, beta) = SBT{u^{-0.3} (1+u)^{-beta}}(s).
# Precomputing g on a (s, beta) grid eliminates the SBT from the hot path.

_TABLE_N_U = 256
_TABLE_U_GRID = jnp.geomspace(1e-5, 100.0, _TABLE_N_U)
_TABLE_N_BETA = 100
_TABLE_BETAS = jnp.linspace(3.0, 13.0, _TABLE_N_BETA)

with _warnings.catch_warnings():
    _warnings.filterwarnings("ignore", message="use backend='jax' if desired")
    _TABLE_SBT = SphericalBessel(_TABLE_U_GRID, nu=0, lowring=True,
                                  backend='jax')

_TABLE_S_GRID = jnp.array(_TABLE_SBT.y)  # output s-grid
_LOG_TABLE_S = jnp.log(_TABLE_S_GRID)


def _build_profile_ft_table():
    """Precompute g(s, beta) on a 2-d grid."""
    profiles = (_TABLE_U_GRID[None, :] ** (-0.3)
                * (1.0 + _TABLE_U_GRID[None, :]) ** (-_TABLE_BETAS[:, None]))
    _, g = _TABLE_SBT(profiles, extrap=True)
    # mcfit SphericalBessel uses unitary Hankel convention, missing sqrt(pi/2)
    g = g * jnp.sqrt(jnp.pi / 2.0)
    return g  # (n_beta, n_s)


_G_TABLE = _build_profile_ft_table()


# ---------------------------------------------------------------------------
# 4c. Arnaud 2010 precomputed 1-D FT lookup table
# ---------------------------------------------------------------------------
# The A10 profile has *all fixed* parameters, so the kernel is:
#   h(u) = u^{-gamma} * (1 + u^alpha)^{(gamma-beta)/alpha}
# and the FT g(s) is a 1-D function of s only (no beta axis).

def _build_a10_ft_table():
    """Precompute g_A10(s) on the shared s-grid (1-D)."""
    kernel = (_TABLE_U_GRID ** (-_A10_GAMMA)
              * (1.0 + _TABLE_U_GRID ** _A10_ALPHA)
              ** ((_A10_GAMMA - _A10_BETA) / _A10_ALPHA))
    _, g = _TABLE_SBT(kernel, extrap=True)
    # mcfit SphericalBessel uses unitary Hankel convention, missing sqrt(pi/2)
    g = g * jnp.sqrt(jnp.pi / 2.0)
    return g  # (n_s,)


_A10_G_TABLE = _build_a10_ft_table()


# ---------------------------------------------------------------------------
# 6. P_Delta — thermal pressure at the virial radius
# ---------------------------------------------------------------------------
def _P200c(M, r_delta, f_b):
    """Thermal pressure P_Delta in eV/cm^3.

    General formula: P_Delta = 3 G M^2 f_b / (8 pi r^4)
    Works for any overdensity definition (200c, 500c, etc.)
    """
    M_kg = M * _Msun_kg
    r_m  = r_delta * MPC_TO_M
    P_SI = 3.0 * _G_SI * M_kg**2 * f_b / (8.0 * jnp.pi * r_m**4)
    P_eV_cm3 = P_SI / Const._eV_ * 1e-6   # Pa -> eV/cm^3
    return P_eV_cm3


def _P500c_arnaud10(M_h, z, h, Ez, alpha_p=0.12):
    """Arnaud 2010 electron pressure normalization P_e,500 in eV/cm^3.

    From Arnaud et al. 2010 Eq. 12, with the CLASS-SZ (0.7/h)^1.5 factor.
    P0 is NOT included here (it enters through the profile FT).

    Parameters
    ----------
    M_h : array — M_500c in M_sun/h
    z   : array — redshift (unused, kept for API clarity)
    h   : float — H0/100
    Ez  : array — E(z) = H(z)/H0
    alpha_p : float — mass slope correction (default 0.12)

    Returns
    -------
    P_e_500 : array — electron pressure in eV/cm^3 (without P0)
    """
    C_pressure = (1.65 * (h / 0.7) ** 2 * Ez ** (8.0 / 3.0)
                  * (M_h / (3.0e14 * 0.7)) ** (2.0 / 3.0 + alpha_p))
    return C_pressure * (0.7 / h) ** 1.5


# ---------------------------------------------------------------------------
# 7. cl_yy — full halo-model C_ell^yy  (optimised: table lookup, no SBT)
# ---------------------------------------------------------------------------
def cl_yy(ell, params, z_grid=None, lnm_grid=None,
          pk_grid=None, k_grid=None,
          Hz=None, chi_z=None,
          sigma_grid=None, dsigma2_grid=None, R_grid=None,
          return_1h_2h=False, profile='battaglia12',
          delta_mean=None):
    """Compute the tSZ angular power spectrum C_ell^yy (1-halo + 2-halo).

    Uses a precomputed profile-FT lookup table for speed.

    Parameters
    ----------
    ell   : 1-d array of multipoles
    params : dict with at least {omega_b, omega_cdm, H0}
    z_grid : 1-d array, shape (n_z,)
    lnm_grid : 1-d array, shape (n_m,) — ln(M) in M_sun/h
    pk_grid : 2-d array, shape (n_k, n_z) — linear P(k,z) in Mpc^3
    k_grid  : 1-d array, shape (n_k,) — wavenumbers in 1/Mpc
    Hz     : 1-d array, shape (n_z,) — H(z)/c in 1/Mpc
    chi_z  : 1-d array, shape (n_z,) — comoving distance in Mpc
    sigma_grid  : 2-d array, shape (n_z, n_m) — sigma(M, z)
    dsigma2_grid : 2-d array, shape (n_z, n_m) — d(sigma^2)/dR
    R_grid : 1-d array, shape (n_m,) — Lagrangian radii in Mpc/h
    return_1h_2h : bool
        If True, return (cl_1h, cl_2h) instead of their sum.
    profile : str
        Pressure profile model: ``'battaglia12'`` (default) or ``'arnaud10'``.

    Returns
    -------
    cl : 1-d array, shape (n_ell,) — C_ell^yy (dimensionless)
        If ``return_1h_2h`` is True, returns ``(cl_1h, cl_2h)`` instead.
    """
    h = params['H0'] / 100.0
    Omega_b = params['omega_b'] / h**2
    Omega_cdm = params['omega_cdm'] / h**2
    Omega_ncdm = params.get('m_ncdm', 0.06) / (93.14 * h**2)
    Omega_m = Omega_b + Omega_cdm + Omega_ncdm
    f_b = Omega_b / Omega_m

    n_z = z_grid.shape[0]
    n_m = lnm_grid.shape[0]
    n_ell = ell.shape[0]

    # Mass grid: lnm_grid is in ln(M [Msun/h]).
    M_grid_h = jnp.exp(lnm_grid)   # M_sun/h
    M_grid = M_grid_h / h           # M_sun (h-free)

    # rho_crit(z) in M_sun / Mpc^3
    Hz_si = Hz * _c_SI / MPC_TO_M  # 1/s
    rho_crit_z = 3.0 * Hz_si**2 / (8.0 * jnp.pi * _G_SI) * MPC_TO_M**3 / _Msun_kg

    # Omega_m(z) for converting Delta_crit -> Delta_mean
    H0_si = params['H0'] * 1.0e3 / MPC_TO_M          # 1/s
    rho_crit_0 = (3.0 * H0_si ** 2
                  / (8.0 * jnp.pi * _G_SI)
                  * MPC_TO_M ** 3 / _Msun_kg)         # M_sun/Mpc^3
    Omega_m_z = (Omega_m * rho_crit_0
                 * (1.0 + z_grid) ** 3 / rho_crit_z)  # (n_z,)

    # If caller did not pass delta_mean, set it from the profile type.
    # CLASS-SZ uses T08M500c for A10 and T08M200c for B12, meaning the
    # Tinker HMF is evaluated at Delta_crit expressed in mean-density units:
    #   Delta_mean(z) = Delta_crit / Omega_m(z)
    if delta_mean is None:
        if profile == 'arnaud10':
            delta_mean = 500.0 / Omega_m_z  # (n_z,)
        else:
            delta_mean = 200.0 / Omega_m_z  # (n_z,)

    if profile == 'arnaud10':
        # --- Arnaud 2010 profile (fixed gNFW, 1-D FT table) ---
        # Mass variable IS M_500c directly (no conversion needed).
        # r_500c  [Mpc], shape (n_z, n_m)
        r_500c = jnp.power(3.0 * M_grid[None, :] / (4.0 * jnp.pi * 500.0
                  * rho_crit_z[:, None]), 1.0 / 3.0)

        # s(ell, z, M) = (ell + 0.5) * r_500c * (1+z) / (c500 * chi)
        # The (1+z) converts chi to d_A: ell_s = d_A/r_s.
        s_query = ((ell[:, None, None] + 0.5)
                   * r_500c[None, :, :]
                   * (1.0 + z_grid)[None, :, None]
                   / (_A10_C500 * chi_z[None, :, None]))  # (n_ell, n_z, n_m)

        # 1-D linear interpolation in log(s) space
        n_s = _TABLE_S_GRID.shape[0]
        log_sq = jnp.log(jnp.clip(s_query, 1e-30)).ravel()
        is_ = jnp.searchsorted(_LOG_TABLE_S, log_sq) - 1
        is_ = jnp.clip(is_, 0, n_s - 2)
        ts = ((log_sq - _LOG_TABLE_S[is_])
              / (_LOG_TABLE_S[is_ + 1] - _LOG_TABLE_S[is_]))
        g_interp = (_A10_G_TABLE[is_] * (1.0 - ts)
                    + _A10_G_TABLE[is_ + 1] * ts)
        g_interp = g_interp.reshape(n_ell, n_z, n_m)

        # u_tilde = 4 pi * P0 * (r_500c / c500)^3 * g(s)   [Mpc^3]
        r_over_c = r_500c / _A10_C500
        u_at_ell = (4.0 * jnp.pi * _A10_P0
                    * r_over_c[None, :, :] ** 3
                    * g_interp)  # (n_ell, n_z, n_m)

        # P_500c for each (z, M)
        P_delta_grid = _P200c(M_grid[None, :], r_500c, f_b)  # (n_z, n_m)

    else:
        # --- Battaglia 2012 profile (2-D FT table) ---
        # Mass variable IS M_200c directly (no conversion needed).
        # r_200c(M, z)  [Mpc], shape (n_z, n_m)
        r_200c = jnp.power(3.0 * M_grid[None, :] / (4.0 * jnp.pi * 200.0
                  * rho_crit_z[:, None]), 1.0 / 3.0)

        # Battaglia profile parameters via broadcasting
        P0, xc, beta_vals = battaglia12_profile_params(
            M_grid[None, :], z_grid[:, None])

        # s(ell, z, M) = (ell + 0.5) * r_200c * xc * (1+z) / chi
        # The (1+z) converts chi to d_A: ell_s = d_A/r_s.
        s_query = ((ell[:, None, None] + 0.5)
                   * r_200c[None, :, :] * xc[None, :, :]
                   * (1.0 + z_grid)[None, :, None]
                   / chi_z[None, :, None])  # (n_ell, n_z, n_m)

        # Bilinear interpolation in (log_s, beta) space
        n_s = _TABLE_S_GRID.shape[0]
        n_beta = _TABLE_BETAS.shape[0]

        beta_flat = beta_vals.ravel()
        ib = jnp.searchsorted(_TABLE_BETAS, beta_flat) - 1
        ib = jnp.clip(ib, 0, n_beta - 2)
        tb = (beta_flat - _TABLE_BETAS[ib]) / (_TABLE_BETAS[ib + 1] - _TABLE_BETAS[ib])

        ib_b = jnp.tile(ib, n_ell)
        tb_b = jnp.tile(tb, n_ell)

        log_sq = jnp.log(jnp.clip(s_query, 1e-30)).ravel()
        is_ = jnp.searchsorted(_LOG_TABLE_S, log_sq) - 1
        is_ = jnp.clip(is_, 0, n_s - 2)
        ts = (log_sq - _LOG_TABLE_S[is_]) / (_LOG_TABLE_S[is_ + 1] - _LOG_TABLE_S[is_])

        g00 = _G_TABLE[ib_b, is_]
        g01 = _G_TABLE[ib_b, is_ + 1]
        g10 = _G_TABLE[ib_b + 1, is_]
        g11 = _G_TABLE[ib_b + 1, is_ + 1]
        g_interp = ((1 - tb_b) * (1 - ts) * g00 + (1 - tb_b) * ts * g01
                    + tb_b * (1 - ts) * g10 + tb_b * ts * g11)
        g_interp = g_interp.reshape(n_ell, n_z, n_m)

        # u_tilde(k_ell) = 4 pi r_200c^3 P0 xc^3 g(s, beta)   [Mpc^3]
        u_at_ell = (4.0 * jnp.pi * r_200c[None, :, :] ** 3
                    * P0[None, :, :] * xc[None, :, :] ** 3
                    * g_interp)  # (n_ell, n_z, n_m)

        # P_200c for each (z, M)
        P_delta_grid = _P200c(M_grid[None, :], r_200c, f_b)  # (n_z, n_m)

    # --- y_ell = PREFAC_Y * P_delta * u_tilde * (1+z)^2 / chi^2 ---
    onepz2_chi2 = (1.0 + z_grid) ** 2 / chi_z ** 2  # (n_z,)
    y_ell = (PREFAC_Y * P_delta_grid[None, :, :] * u_at_ell
             * onepz2_chi2[None, :, None])  # (n_ell, n_z, n_m)

    # --- HMF: dn/dlnM ---
    hmf_f = tinker08_hmf(sigma_grid, z_grid, delta_mean)  # (n_z, n_m)

    Rh = R_grid  # R_grid is already in Mpc/h
    lnSigma2 = 2.0 * jnp.log(sigma_grid)
    dlnSigma2dlnR = dsigma2_grid * R_grid[None, :] / jnp.exp(lnSigma2)
    dlnnudlnRh = -dlnSigma2dlnR  # CLASS-SZ convention: -dlnσ²/dlnR (factor 0.5 is in tinker08_hmf)

    dndlnm_h = ((1.0 / 3.0) * 3.0 / (4.0 * jnp.pi * Rh[None, :] ** 3)
                * jnp.abs(dlnnudlnRh) * hmf_f)  # h^3/Mpc^3
    dndlnm = dndlnm_h * h ** 3  # 1/Mpc^3 (h-free)

    # --- Bias (vectorised, no vmap) ---
    nu_grid = DELTA_C / sigma_grid  # (n_z, n_m)
    bias_grid = tinker10_bias(nu_grid, delta_mean[:, None])  # (n_z, n_m)

    # --- Volume element ---
    dVdzdOmega = chi_z ** 2 / Hz  # (n_z,)

    # --- 1-halo: C_ell^1h = int dz dV * int dlnM dn/dlnM * y_ell^2 ---
    integ_1h = dndlnm[None, :, :] * y_ell ** 2  # (n_ell, n_z, n_m)
    I_m_1h = jnp.trapezoid(integ_1h, lnm_grid, axis=2)  # (n_ell, n_z)
    cl_1h = jnp.trapezoid(dVdzdOmega[None, :] * I_m_1h, z_grid, axis=1)

    # --- 2-halo ---
    integ_2h = dndlnm[None, :, :] * bias_grid[None, :, :] * y_ell
    I_m_2h = jnp.trapezoid(integ_2h, lnm_grid, axis=2)  # (n_ell, n_z)

    # P_lin(k_ell, z) via vectorised searchsorted
    log_k_pk = jnp.log(k_grid)
    log_pk = jnp.log(pk_grid)  # (n_k, n_z)
    k_ell_z = (ell[:, None] + 0.5) / chi_z[None, :]  # (n_ell, n_z)
    log_kq = jnp.log(k_ell_z).ravel()  # (n_ell * n_z,)

    ik = jnp.searchsorted(log_k_pk, log_kq) - 1
    ik = jnp.clip(ik, 0, k_grid.shape[0] - 2)
    tk = ((log_kq - log_k_pk[ik])
          / (log_k_pk[ik + 1] - log_k_pk[ik]))
    z_flat = jnp.tile(jnp.arange(n_z), n_ell)
    pk_lo = log_pk[ik, z_flat]
    pk_hi = log_pk[ik + 1, z_flat]
    pk_at_ell = jnp.exp(pk_lo + tk * (pk_hi - pk_lo)).reshape(n_ell, n_z)

    cl_2h = jnp.trapezoid(
        dVdzdOmega[None, :] * pk_at_ell * I_m_2h ** 2, z_grid, axis=1)

    if return_1h_2h:
        return cl_1h, cl_2h
    return cl_1h + cl_2h


# ---------------------------------------------------------------------------
# 8. prepare_grids — build all inputs for cl_yy from CosmoPower emulators
# ---------------------------------------------------------------------------
def prepare_grids(params, z_grid=None, n_z=100,
                  m_min_h=1e10, m_max_h=5e16, cosmo_model='lcdm',
                  match_class_sz=False):
    """Prepare input grids for cl_yy using CosmoPower JAX emulators.

    Evaluates P(k,z), H(z), chi(z) via the CosmoPower emulators in
    classy_szfast, and sigma(R,z) via mcfit TophatVar.

    Parameters
    ----------
    params : dict
        Cosmological parameters. Must include at least:
        omega_b, omega_cdm, H0, ``ln10^{10}A_s``, n_s.
    z_grid : 1-d array, optional
        Redshift grid. Default: ``geomspace(5e-3, 5.0, n_z)``.
    n_z : int
        Number of z points (used only if z_grid is None).
    m_min_h, m_max_h : float
        Mass range in M_sun/h.
    cosmo_model : str
        Cosmological model label for the emulators (default 'lcdm').
    match_class_sz : bool
        If True, reproduce the CLASS-SZ fast-mode R-unit convention:
        sigma and dsigma2/dR are evaluated at R_Mpc = R_h/h instead of
        R_h, and the derivative is scaled by 1/h so that the effective
        dlnσ²/dlnR uses R_Mpc.  Default False (physically correct).

    Returns
    -------
    grids : dict
        All keyword arguments accepted by ``cl_yy``.
    """
    import numpy as np
    from mcfit import TophatVar
    from .cosmopower_jax import cp_pkl_nn_jax, cp_h_nn_jax, cp_da_nn_jax
    from .emulators_meta_data import emulator_dict

    h = params['H0'] / 100.0

    # Merge caller params with emulator defaults
    full = dict(emulator_dict[cosmo_model]['default'])
    full.update(params)

    # ---- z-grid --------------------------------------------------------
    if z_grid is None:
        z_grid = jnp.geomspace(5e-3, 5.0, n_z)
    z_grid = jnp.where(z_grid < 5e-3, 5e-3, z_grid)
    n_z_val = int(z_grid.shape[0])

    # ---- k-grid (h/Mpc) and PK conversion factor ----------------------
    ks_h = jnp.geomspace(1e-4, 50.0, 5000)[::10]        # 500 pts, h/Mpc
    ls_pk = jnp.arange(2, 5002)[::10]
    pk_power_fac = 1.0 / (ls_pk * (ls_pk + 1.0) / (2.0 * jnp.pi))

    # ---- P(k,z) via CosmoPower pkl emulator ----------------------------
    pkl_em = cp_pkl_nn_jax[cosmo_model]
    params_pk = {k: [v] * n_z_val for k, v in full.items()}
    params_pk['z_pk_save_nonclass'] = [float(z) for z in z_grid]

    log10pk = pkl_em.predict(params_pk)                   # (n_z, 500)
    pk_h = 10.0**log10pk * pk_power_fac                   # (Mpc/h)^3

    # Convert to h-free units
    ks = ks_h * h                                         # 1/Mpc
    pk_grid = (pk_h / h**3).T                             # (n_k, n_z) Mpc^3

    # ---- H(z)/c in 1/Mpc via CosmoPower h emulator --------------------
    z_interp = jnp.linspace(0.0, 20.0, 5000)
    h_em = cp_h_nn_jax[cosmo_model]
    params_h = {k: [v] for k, v in full.items()}
    Hz_interp = h_em.predict(params_h)                    # (5000,) H/c [1/Mpc]
    Hz = jnp.interp(z_grid, z_interp, Hz_interp)

    # ---- chi(z) in Mpc via CosmoPower da emulator ----------------------
    da_em = cp_da_nn_jax[cosmo_model]
    params_da = {k: [v] for k, v in full.items()}
    dA_vals = da_em.predict(params_da)                    # (5000,) d_A [Mpc]
    chi_vals = dA_vals * (1.0 + z_interp)                 # chi = d_A*(1+z)
    chi_z = jnp.interp(z_grid, z_interp, chi_vals)
    chi_z = jnp.where(chi_z < 1.0, 1.0, chi_z)

    # ---- sigma(R,z) and dsigma^2/dR via TophatVar (h-units) -----------
    # Batched: pk_h is (n_z, 500), TophatVar operates on last axis
    tv_var = TophatVar(ks_h, lowring=True, backend='jax')
    R_h_full, var_full = tv_var(pk_h, extrap=True)        # (n_z, n_R)
    R_h_full = R_h_full.flatten()                         # Mpc/h

    sigma_full = jnp.sqrt(var_full)                       # (n_z, n_R)

    # dsigma^2/dR via numerical gradient (matching classy_szfast.py)
    R_h_np = np.array(R_h_full)
    var_np = np.array(var_full)
    dvar_np = np.empty_like(var_np)
    for iz in range(n_z_val):
        dvar_np[iz, :] = np.gradient(var_np[iz, :], R_h_np)
    dvar_full = jnp.array(dvar_np)

    # ---- Mass from R; filter to [m_min_h, m_max_h] --------------------
    Omega_cb = full['omega_b'] / h**2 + full['omega_cdm'] / h**2
    H0_code = full['H0'] * 1e3 / _c_SI
    Rho_crit_0 = ((3.0 / (8.0 * jnp.pi * _G_SI * _Msun_kg))
                  * (Const._Mpc_over_m_) * _c_SI**2
                  * H0_code**2 / h**2)
    M_h_full = (4.0 / 3.0) * jnp.pi * Omega_cb * Rho_crit_0 * R_h_full**3

    if m_min_h is not None and m_max_h is not None:
        mask = np.array((M_h_full >= m_min_h) & (M_h_full <= m_max_h))
        idx = np.where(mask)[0]
    else:
        idx = np.arange(len(M_h_full))

    # --- Optionally reproduce CLASS-SZ fast-mode R-unit convention ----------
    # CLASS-SZ stores sigma(R_h) in Mpc/h from TophatVar, but queries at
    # R_asked = R_h/h (Mpc).  This shifts the lookup to a larger radius.
    # The derivative dlnσ²/dlnR also uses R_Mpc = R_h/h internally.
    if match_class_sz:
        lnR_full_np = np.log(R_h_np)                      # ln(R) in Mpc/h
        R_masked_np = np.array(R_h_full[idx])
        lnR_shifted = np.log(R_masked_np / h)              # ln(R_Mpc)

        sigma_np = np.array(sigma_full)
        sigma_shifted = np.empty((n_z_val, len(idx)))
        dsigma2_shifted = np.empty((n_z_val, len(idx)))
        for iz in range(n_z_val):
            lnsig = np.log(sigma_np[iz, :])
            sigma_shifted[iz, :] = np.exp(
                np.interp(lnR_shifted, lnR_full_np, lnsig))
            dsigma2_shifted[iz, :] = np.interp(
                lnR_shifted, lnR_full_np, dvar_np[iz, :])

        # cl_yy computes: dlnσ²/dlnR = dsigma2 * R_h / σ²
        # CLASS-SZ uses:               dsigma2(R_Mpc) * R_Mpc / σ²
        # Since R_Mpc = R_h/h, we scale dsigma2 by 1/h so that
        # dsigma2_eff * R_h = dsigma2(R_Mpc) * R_Mpc.
        dsigma2_shifted /= h

        sigma_out = jnp.array(sigma_shifted)
        dsigma2_out = jnp.array(dsigma2_shifted)
    else:
        sigma_out = sigma_full[:, idx]
        dsigma2_out = dvar_full[:, idx]

    return dict(
        params=params,
        z_grid=z_grid,
        lnm_grid=jnp.log(M_h_full[idx]),
        pk_grid=pk_grid,
        k_grid=ks,
        Hz=Hz,
        chi_z=chi_z,
        sigma_grid=sigma_out,
        dsigma2_grid=dsigma2_out,
        R_grid=R_h_full[idx],                             # Mpc/h
    )
