"""
Throughput benchmark: loop over random cosmo + profile params.

Draws N random parameter sets from a reasonable prior volume,
evaluates cl_yy_from_params for each, and reports timing statistics.

Run:  python notebooks/benchmark_throughput.py
"""
import warnings; warnings.filterwarnings('ignore')
import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
import time

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from classy_szfast.differentiable import (
    CosmoParams, ProfileParamsA10, cl_yy_from_params,
)

# ── Config ────────────────────────────────────────────────────────────

N_SAMPLES = 200
ell = jnp.geomspace(2, 5000, 80)

# Parameter ranges: (min, max) — uniform draws
COSMO_RANGES = {
    'omega_b':      (0.019, 0.025),
    'omega_cdm':    (0.10, 0.14),
    'H0':           (60.0, 80.0),
    'tau_reio':     (0.04, 0.08),
    'ln10_10_As':   (2.9, 3.2),
    'n_s':          (0.92, 1.01),
}

PROFILE_RANGES = {
    'P0':    (5.0, 12.0),
    'c500':  (0.8, 1.6),
    'gamma': (0.1, 0.6),
    'alpha': (0.8, 1.4),
    'beta':  (4.0, 7.0),
}

# ── Draw random samples ──────────────────────────────────────────────

rng = np.random.default_rng(42)

cosmo_samples = []
profile_samples = []
for _ in range(N_SAMPLES):
    c = {k: rng.uniform(*v) for k, v in COSMO_RANGES.items()}
    p = {k: rng.uniform(*v) for k, v in PROFILE_RANGES.items()}
    cosmo_samples.append(CosmoParams(**c))
    profile_samples.append(ProfileParamsA10(**p))

# ── Warm-up (first call includes JIT / emulator load) ────────────────

print(f"Warming up (1 evaluation) ...")
t0 = time.perf_counter()
cl_yy_from_params(ell, cosmo_samples[0], profile_params=profile_samples[0],
                  profile='arnaud10', delta_crit=500.0)
t_warmup = time.perf_counter() - t0
print(f"  Warm-up: {t_warmup*1e3:.0f} ms\n")

# ── Timed loop ────────────────────────────────────────────────────────

print(f"Running {N_SAMPLES} evaluations ...")
times = []
for i, (c, p) in enumerate(zip(cosmo_samples, profile_samples)):
    t0 = time.perf_counter()
    cl_1h, cl_2h = cl_yy_from_params(
        ell, c, profile_params=p,
        profile='arnaud10', delta_crit=500.0,
        n_z=100, n_m=200,
    )
    cl_1h.block_until_ready()
    cl_2h.block_until_ready()
    dt = time.perf_counter() - t0
    times.append(dt)

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_SAMPLES}  last={dt*1e3:.1f} ms  "
              f"running mean={np.mean(times)*1e3:.1f} ms")

times = np.array(times) * 1e3  # convert to ms

# ── Report ────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Forward pass: {N_SAMPLES} random (cosmo, profile) pairs")
print(f"{'='*55}")
print(f"  Mean:     {times.mean():.1f} ms")
print(f"  Median:   {np.median(times):.1f} ms")
print(f"  Std:      {times.std():.1f} ms")
print(f"  Min:      {times.min():.1f} ms")
print(f"  Max:      {times.max():.1f} ms")
print(f"  p5/p95:   {np.percentile(times,5):.1f} / {np.percentile(times,95):.1f} ms")
print(f"  Total:    {times.sum()/1e3:.1f} s")
print(f"  Throughput: {1e3/times.mean():.1f} evals/s")
print(f"{'='*55}")

# ── Also time gradient evaluations ───────────────────────────────────

N_GRAD = 50
print(f"\nRunning {N_GRAD} gradient evaluations ...")

def loss(c, p):
    cl_1h, cl_2h = cl_yy_from_params(
        ell, c, profile_params=p,
        profile='arnaud10', delta_crit=500.0,
        n_z=100, n_m=200,
    )
    return jnp.sum(cl_1h + cl_2h)

grad_fn = jax.grad(loss, argnums=(0, 1))

# Warm up grad
print("  Gradient warm-up ...")
t0 = time.perf_counter()
g_c, g_p = grad_fn(cosmo_samples[0], profile_samples[0])
_ = g_c.omega_b.block_until_ready()
print(f"  Warm-up: {time.perf_counter()-t0:.1f} s\n")

grad_times = []
for i in range(N_GRAD):
    c, p = cosmo_samples[i], profile_samples[i]
    t0 = time.perf_counter()
    g_c, g_p = grad_fn(c, p)
    _ = g_c.omega_b.block_until_ready()
    dt = time.perf_counter() - t0
    grad_times.append(dt)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{N_GRAD}  last={dt*1e3:.0f} ms  "
              f"running mean={np.mean(grad_times)*1e3:.0f} ms")

grad_times = np.array(grad_times) * 1e3

print(f"\n{'='*55}")
print(f"  Gradient (cosmo+profile): {N_GRAD} random pairs")
print(f"{'='*55}")
print(f"  Mean:     {grad_times.mean():.0f} ms")
print(f"  Median:   {np.median(grad_times):.0f} ms")
print(f"  Std:      {grad_times.std():.0f} ms")
print(f"  Min:      {grad_times.min():.0f} ms")
print(f"  Max:      {grad_times.max():.0f} ms")
print(f"  p5/p95:   {np.percentile(grad_times,5):.0f} / {np.percentile(grad_times,95):.0f} ms")
print(f"  Total:    {grad_times.sum()/1e3:.1f} s")
print(f"  Throughput: {1e3/grad_times.mean():.2f} grads/s")
print(f"  Overhead vs fwd: {grad_times.mean()/times.mean():.1f}x")
print(f"{'='*55}")
