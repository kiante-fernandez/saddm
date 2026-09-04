"""
Parameter recovery study for the DDM-SA model using NUTS.

Generates 100 parameter configurations via Latin Hypercube Sampling,
simulates 500 trials per configuration, fits each with PyMC (NUTS),
and evaluates recovery quality.

"""

import os
import time
import warnings

import numpyro

numpyro.set_host_device_count(2)

import numpy as np
import arviz as az

from saddm import make_ddmsa_model, sample_ddmsa, sample_ddmsa_exact

# =====================================================================
# Configuration
# =====================================================================

N_CONFIGS = 100
N_TRIALS = 500
N_CHAINS = 4
N_DRAWS = 1000
N_TUNE = 2000
TARGET_ACCEPT = 0.90
NUTS_BACKEND = "numpyro"   # ~4x the ESS/s of the default C backend

PARAM_RANGES = {
    'a':      (0.65, 2.40),   # boundary separation
    'z':      (0.30, 0.70),   # relative starting point
    'v':      (-3.20, 3.20),  # drift rate
    't':      (0.15, 0.50),   # non-decision time (seconds)
    'sv':     (0.10, 3.00),   # drift rate variability (Gaussian SD)
    'sa_frac': (0.05, 0.90),  # boundary variability as a fraction of a
    'st':     (0.03, 0.15),   # non-decision time variability (seconds)
}

GRID_NAMES = list(PARAM_RANGES.keys())          # sampled by Latin Hypercube
PARAM_NAMES = ['a', 'z', 'v', 't', 'sv', 'sa', 'st']   # reported / recovered

RESULTS_DIR = os.environ.get('OUT_DIR', os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'recovery'))
SHARD = int(os.environ.get('SHARD', '-1'))
N_SHARDS = int(os.environ.get('N_SHARDS', '1'))
_SUFFIX = f'_shard{SHARD}' if SHARD >= 0 else ''
RESULTS_CSV = os.path.join(RESULTS_DIR, f'ddm_sa_recovery_nuts{_SUFFIX}.csv')
FAILURES_LOG = os.path.join(RESULTS_DIR, f'failures{_SUFFIX}.log')


# =====================================================================
# Latin Hypercube Sampling
# =====================================================================

def generate_parameter_grid(n=N_CONFIGS, seed=2024):
    """Generate n parameter configurations using Latin Hypercube Sampling."""
    from scipy.stats.qmc import LatinHypercube

    sampler = LatinHypercube(d=len(GRID_NAMES), seed=seed)
    samples = sampler.random(n=n)

    configs = []
    for i in range(n):
        cfg = {'config_id': i}
        for j, name in enumerate(GRID_NAMES):
            lo, hi = PARAM_RANGES[name]
            cfg[name] = lo + samples[i, j] * (hi - lo)
        cfg['sa'] = cfg.pop('sa_frac') * cfg['a']
        configs.append(cfg)

    return configs


# =====================================================================
# PyMC Model
# =====================================================================

def build_model(data):
    """Build the PyMC model. Thin wrapper over saddm.ddmsa.make_ddmsa_model."""
    return make_ddmsa_model(data, use_potential=True)


# =====================================================================
# Fitting and Extraction
# =====================================================================

def fit_and_extract(cfg, n_trials=N_TRIALS):
    """Simulate data, fit model, and extract posterior summaries."""
    data = sample_ddmsa_exact(
        a=cfg['a'], z=cfg['z'], v=cfg['v'], t=cfg['t'],
        sv=cfg['sv'], sa=cfg['sa'], st=cfg['st'],
        n_trials=n_trials, seed=cfg['config_id']
    )

    if len(data) < 50:
        raise ValueError(f"Too few valid trials: {len(data)}")

    mean_rt = np.mean(data[:, 0])
    accuracy = np.mean(data[:, 1])
    print(f"  Simulated: {len(data)} trials, mean RT={mean_rt:.3f}s, "
          f"accuracy(upper)={accuracy:.2%}")

    model = build_model(data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = sample_ddmsa(
            model,
            backend=NUTS_BACKEND,
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            target_accept=TARGET_ACCEPT,
            progressbar=False,
            random_seed=cfg['config_id'],
        )

    # Extract results
    result = {'config_id': cfg['config_id']}

    for param in PARAM_NAMES:
        result[f'true_{param}'] = cfg[param]

        samples = trace.posterior[param].values.flatten()
        result[f'post_mean_{param}'] = np.mean(samples)
        result[f'post_median_{param}'] = np.median(samples)
        result[f'post_std_{param}'] = np.std(samples)

        hdi = az.hdi(trace, var_names=[param], hdi_prob=0.95)
        hdi_vals = hdi[param].values
        result[f'hdi_lo_{param}'] = float(hdi_vals[0])
        result[f'hdi_hi_{param}'] = float(hdi_vals[1])

        true_val = cfg[param]
        result[f'coverage_{param}'] = int(
            hdi_vals[0] <= true_val <= hdi_vals[1]
        )

    rhat = az.rhat(trace)
    for param in PARAM_NAMES:
        result[f'rhat_{param}'] = float(rhat[param].values)

    # Check for divergences
    if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
        result['n_divergences'] = int(trace.sample_stats['diverging'].values.sum())
    else:
        result['n_divergences'] = 0

    result['n_trials'] = len(data)
    result['mean_rt'] = mean_rt
    result['accuracy'] = accuracy

    return result


# =====================================================================
# Recovery Study with Checkpointing
# =====================================================================

def run_recovery_study():
    """Run the full parameter recovery study with CSV checkpointing."""
    import pandas as pd

    os.makedirs(RESULTS_DIR, exist_ok=True)

    configs = generate_parameter_grid(n=N_CONFIGS, seed=2024)
    if SHARD >= 0:
        configs = [c for c in configs if c['config_id'] % N_SHARDS == SHARD]

    completed_ids = set()
    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        completed_ids = set(existing['config_id'].values)
        print(f"Found {len(completed_ids)} completed configurations.")

    remaining = [c for c in configs if c['config_id'] not in completed_ids]
    print(f"Running {len(remaining)} of {len(configs)} configurations.")

    for i, cfg in enumerate(remaining):
        cfg_id = cfg['config_id']
        print(f"\n{'='*60}")
        print(f"Config {cfg_id} ({i+1}/{len(remaining)})")
        print(f"  True: a={cfg['a']:.3f} z={cfg['z']:.3f} v={cfg['v']:.3f} "
              f"t={cfg['t']:.3f} sv={cfg['sv']:.3f} sa={cfg['sa']:.3f} "
              f"st={cfg['st']:.3f}")

        t0 = time.time()
        try:
            result = fit_and_extract(cfg)
            elapsed = time.time() - t0
            result['elapsed_seconds'] = elapsed

            row_df = pd.DataFrame([result])
            header = not os.path.exists(RESULTS_CSV)
            row_df.to_csv(RESULTS_CSV, mode='a', header=header, index=False)

            print(f"  Elapsed: {elapsed:.1f}s  Divergences: {result['n_divergences']}")
            for param in PARAM_NAMES:
                true_v = result[f'true_{param}']
                post_v = result[f'post_median_{param}']
                rhat_v = result[f'rhat_{param}']
                cov = result[f'coverage_{param}']
                print(f"    {param:3s}: true={true_v:.4f} "
                      f"post={post_v:.4f} rhat={rhat_v:.3f} "
                      f"cov={'Y' if cov else 'N'}")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            with open(FAILURES_LOG, 'a') as f:
                f.write(f"config_id={cfg_id} error={e}\n")

    print(f"\nResults saved to {RESULTS_CSV}")


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == '__main__':
    run_recovery_study()
