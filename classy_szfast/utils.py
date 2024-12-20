import numpy as np
from datetime import datetime
import multiprocessing
import time
import functools
import re
from pkg_resources import resource_filename
import os
from scipy import optimize
from scipy.integrate import quad
from scipy.interpolate import interp1d
import math
from numpy import linalg as LA
import mcfit
from mcfit import P2xi
import jax
import jax.numpy as jnp
# import cosmopower
# import classy_sz as csz



from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator

kb = 1.38064852e-23 #m2 kg s-2 K-1
clight = 299792458. #m/s
hplanck=6.62607004e-34 #m2 kg / s
firas_T0 = 2.728 #pivot temperature used in the Max Lkl Analysis
firas_T0_bf = 2.725 #best-fitting temperature

Tcmb_uk = 2.7255e6

G_newton = 6.674e-11
rho_crit_over_h2_in_GeV_per_cm3 = 1.0537e-5


nu_21_cm_in_GHz =  1./21.1*clight*1.e2/1.e9
x_21_cm = hplanck*nu_21_cm_in_GHz/kb/firas_T0_bf*1.e9

kappa_c = 2.1419 # 4M_2-3M_c see below eq. 9b of https://arxiv.org/pdf/1506.06582.pdf

beta_mu = 2.1923

G1 = np.pi**2./6
G2 = 2.4041
G3 = np.pi**4/15.
a_rho = G2/G3
alpha_mu = 2.*G1/3./G2 # = 1/beta_mu = π^2/18ζ(3) see eq. 4.15 CUSO lectures.

z_mu_era = 3e5
z_y_era = 5e4
z_reio_min = 6
z_reio_max = 25
z_recombination_min = 800
z_recombination_max = 1500

# Physical constants
# ------------------
# Light speed
class Const:
    c_km_s = 299792.458  # speed of light
    h_J_s = 6.626070040e-34  # Planck's constant
    kB_J_K = 1.38064852e-23  # Boltzmann constant

    _c_ = 2.99792458e8      # c in m/s 
    _Mpc_over_m_ = 3.085677581282e22  # conversion factor from meters to megaparsecs 
    _Gyr_over_Mpc_ = 3.06601394e2 # conversion factor from megaparsecs to gigayears
    _G_ = 6.67428e-11             # Newton constant in m^3/Kg/s^2 
    _eV_ = 1.602176487e-19        # 1 eV expressed in J 

    # parameters entering in Stefan-Boltzmann constant sigma_B 
    _k_B_ = 1.3806504e-23
    _h_P_ = 6.62606896e-34
    _M_sun_ =  1.98855e30 # solar mass in kg



def jax_gradient(f, *varargs, axis=None, edge_order=1):
    f = jnp.asarray(f)
    N = f.ndim  # number of dimensions
    if axis is None:
        axes = tuple(range(N))
    else:
        axes = jax.numpy._normalize_axis_index(axis, N)
    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    elif n == 1 and jnp.ndim(varargs[0]) == 0:
        # single scalar for all axes
        dx = varargs * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(varargs)
        for i, distances in enumerate(dx):
            distances = jnp.asarray(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1D")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError("when 1D, distances must match "
                                 "the length of the corresponding dimension")
            if jnp.issubdtype(distances.dtype, jnp.integer):
                # Convert jax integer types to float64 to avoid modular
                # arithmetic in np.diff(distances).
                distances = distances.astype(jnp.float64)
            diffx = jnp.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")
    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")
    outvals = []
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N
    otype = f.dtype
    # All other types convert to floating point.
    if jnp.issubdtype(otype, jnp.integer):
        f = f.astype(jnp.float64)
    otype = jnp.float64
    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required.")
        # result allocation
        out = jnp.empty_like(f, dtype=otype)
        uniform_spacing = jnp.ndim(ax_dx) == 0
        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)
        if uniform_spacing:
            out = out.at[tuple(slice1)].set(
                (f[tuple(slice4)] - f[tuple(slice2)]) / (2.0 * ax_dx)
            )
        else:
            dx1 = ax_dx[:-1]
            dx2 = ax_dx[1:]
            a = -(dx2) / (dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            shape = [1] * N
            shape[axis] = -1
            a = a.reshape(shape)
            b = b.reshape(shape)
            c = c.reshape(shape)
            out = out.at[tuple(slice1)].set(
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )
        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            out = out.at[tuple(slice1)].set(
                (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0
            )
            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            out = out.at[tuple(slice1)].set(
                (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n
            )
        # Numerical differentiation: 2nd order edges
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2. / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2. * dx1 + dx2) / (dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = -dx1 / (dx2 * (dx1 + dx2))
            out = out.at[tuple(slice1)].set(
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )
            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2. / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = dx2 / (dx1 * (dx1 + dx2))
                b = -(dx2 + dx1) / (dx1 * dx2)
                c = (2. * dx2 + dx1) / (dx2 * (dx1 + dx2))
            out = out.at[tuple(slice1)].set(
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )
        outvals.append(out)
        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)
    if len_axes == 1:
        return outvals[0]
    ret = tuple(outvals)
    print("return",ret)