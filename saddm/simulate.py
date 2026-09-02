"""
DDM simulator using Euler-Maruyama integration.

Generates synthetic data with known parameters. Diffusion coefficient s = 1
throughout, matching saddm.ddmsa and saddm.core; there is deliberately no knob to
change it, because the likelihood is written for s = 1 only.

Euler-Maruyama overshoots the boundary by O(sqrt(dt)), which biases mean RT high by
~0.4% at dt=1e-4 and inflates recovered a and sv. For parameter recovery prefer
saddm.ddmsa.sample_ddmsa_exact, which inverts the analytic CDF; keep this simulator
as an independent check on the likelihood.
"""

import numpy as np


def simulate_ddm_trial(a, z, v, ter, sv=0.0, sa=0.0, st=0.0, sz=0.0,
                       dt=0.0001, max_time=10.0, rng=None):
    """Simulate a single DDM trial with across-trial variability at s = 1.

    Parameters:
        a:   boundary separation (~0.65-2.4)
        z:   relative starting point (0 to 1)
        v:   drift rate (~0.1-3.2)
        ter: non-decision time (seconds)
        sv:  Gaussian drift rate variability (SD)
        sa:  uniform boundary separation variability (full width)
        st:  uniform non-decision time variability (full width)
        sz:  uniform starting-point variability (full width, in relative-z units)
        dt:  simulation time step (seconds)
        max_time: maximum decision time before timeout (seconds)
        rng: numpy random generator instance

    Returns:
        (rt, choice) where rt is in seconds and choice is 0 (lower) or 1 (upper).
        Returns (np.nan, -1) on timeout.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample trial-level variability
    trial_v = rng.normal(v, sv) if sv > 0 else v

    trial_a = a
    if sa > 0:
        trial_a = a + rng.uniform(-sa / 2, sa / 2)
        trial_a = max(0.01, trial_a)

    trial_ter = ter
    if st > 0:
        trial_ter = ter + rng.uniform(-st / 2, st / 2)
        trial_ter = max(0.0, trial_ter)

    trial_z = z
    if sz > 0:
        trial_z = z + rng.uniform(-sz / 2, sz / 2)
        trial_z = min(max(1e-4, trial_z), 1.0 - 1e-4)

    # Starting point in absolute units
    x = trial_a * trial_z
    decision_time = 0.0
    sqrt_dt = np.sqrt(dt)

    # Euler-Maruyama diffusion
    while 0 < x < trial_a:
        dx = trial_v * dt + sqrt_dt * rng.standard_normal()
        x += dx
        decision_time += dt
        if decision_time > max_time:
            return np.nan, -1

    rt = trial_ter + decision_time
    choice = 1 if x >= trial_a else 0
    return rt, choice


def simulate_ddm_data(a, z, v, ter, sv=0.0, sa=0.0, st=0.0, sz=0.0,
                      n_trials=500, seed=None, dt=0.0001):
    """Simulate multiple DDM trials.

    Parameters:
        Same as simulate_ddm_trial, plus:
        n_trials: number of trials to simulate
        seed: random seed for reproducibility

    Returns:
        numpy array of shape (n_trials, 2) with columns [rt, choice].
        Timed-out trials are excluded.
    """
    rng = np.random.default_rng(seed)
    results = []

    for _ in range(n_trials):
        rt, choice = simulate_ddm_trial(
            a, z, v, ter, sv=sv, sa=sa, st=st, sz=sz,
            dt=dt, rng=rng
        )
        if not np.isnan(rt):
            results.append((rt, choice))

    return np.array(results)
