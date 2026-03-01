"""
Plot d(D_ell^yy)/d(param) for cosmological and profile parameters.
"""
import warnings; warnings.filterwarnings('ignore')
import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
import time

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classy_szfast.differentiable import (
    CosmoParams, ProfileParamsA10, cl_yy_from_params,
)

# ── Setup ─────────────────────────────────────────────────────────────

cosmo = CosmoParams(
    omega_b=0.02242, omega_cdm=0.11933, H0=67.66,
    tau_reio=0.0561, ln10_10_As=3.047, n_s=0.9665,
)
profile = ProfileParamsA10()
ell = jnp.geomspace(2, 5000, 60)
dl_fac = ell * (ell + 1) / (2 * jnp.pi) * 1e12

# ── Helper: D_ell^yy(total) as a function of one parameter ───────────

def dl_total(cosmo_, profile_):
    cl1, cl2 = cl_yy_from_params(ell, cosmo_, profile_params=profile_,
                                  profile='arnaud10', delta_crit=500.0)
    return dl_fac * (cl1 + cl2)

# ── 1. Jacobian w.r.t. cosmo params ──────────────────────────────────
print("Computing cosmo Jacobian ...")
t0 = time.perf_counter()
jac_cosmo = jax.jacrev(lambda c: dl_total(c, profile))(cosmo)
print(f"  done in {time.perf_counter()-t0:.1f} s")

# ── 2. Jacobian w.r.t. profile params ────────────────────────────────
print("Computing profile Jacobian ...")
t0 = time.perf_counter()
jac_profile = jax.jacrev(lambda pp: dl_total(cosmo, pp))(profile)
print(f"  done in {time.perf_counter()-t0:.1f} s")

# ── 3. Plot ───────────────────────────────────────────────────────────

cosmo_labels = {
    'omega_b':    r'$\omega_b$',
    'omega_cdm':  r'$\omega_{cdm}$',
    'H0':         r'$H_0$',
    'ln10_10_As': r'$\ln(10^{10}A_s)$',
    'n_s':        r'$n_s$',
    'm_ncdm':     r'$m_\nu$',
}
# skip tau_reio (zero gradient)
cosmo_keys = ['omega_b', 'omega_cdm', 'H0', 'ln10_10_As', 'n_s', 'm_ncdm']

profile_labels = {
    'P0':    r'$P_0$',
    'c500':  r'$c_{500}$',
    'gamma': r'$\gamma$',
    'alpha': r'$\alpha$',
    'beta':  r'$\beta$',
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: cosmo
ax = axes[0]
jac_dict = jac_cosmo._asdict()
for key in cosmo_keys:
    g = np.array(jac_dict[key])
    # Normalize: plot d(ln D_ell)/d(ln param) = (param/D_ell) * dD_ell/dparam
    param_val = float(getattr(cosmo, key))
    dl_ref = np.array(dl_total(cosmo, profile))
    dlog = g * param_val / dl_ref
    ax.semilogx(ell, dlog, lw=1.8, label=cosmo_labels[key])

ax.axhline(0, color='k', ls=':', lw=0.5)
ax.set_xlabel(r"$\ell$", fontsize=15)
ax.set_ylabel(r"$\partial \ln D_\ell^{yy} \,/\, \partial \ln \theta$", fontsize=15)
ax.set_title("Cosmological parameters", fontsize=14)
ax.legend(fontsize=10, ncol=2)
ax.grid(alpha=0.2, ls='--')

# Right: profile
ax = axes[1]
jac_dict_p = jac_profile._asdict()
dl_ref = np.array(dl_total(cosmo, profile))
for key in profile_labels:
    g = np.array(jac_dict_p[key])
    param_val = float(getattr(profile, key))
    dlog = g * param_val / dl_ref
    ax.semilogx(ell, dlog, lw=1.8, label=profile_labels[key])

ax.axhline(0, color='k', ls=':', lw=0.5)
ax.set_xlabel(r"$\ell$", fontsize=15)
ax.set_ylabel(r"$\partial \ln D_\ell^{yy} \,/\, \partial \ln \theta$", fontsize=15)
ax.set_title("Arnaud 2010 profile parameters", fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.2, ls='--')

for ax in axes:
    ax.tick_params(direction='in', which='both')

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__) or '.', 'plot_gradients.png')
plt.savefig(outpath, dpi=150)
print(f"\nPlot saved: {outpath}")
