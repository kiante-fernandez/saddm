from numba import njit, float64
from math import pi, sqrt, exp, log, sin, fabs, ceil, floor

@njit(float64(float64, float64, float64), cache=True)
def ftt_01w(tt, w, err):
    """Compute f(t|0,1,w) following Navarro & Fuss, 2009."""
    # calculate number of terms needed for large t
    if pi * tt * err < 1:
        kl = sqrt(-2 * log(pi * tt * err) / (pi * pi * tt))
        kl = max(kl, 1. / (pi * sqrt(tt)))
    else:
        kl = 1. / (pi * sqrt(tt))

    # calculate number of terms needed for small t
    if 2 * sqrt(2 * pi * tt) * err < 1:
        ks = 2 + sqrt(-2 * tt * log(2 * sqrt(2 * pi * tt) * err))
        ks = max(ks, sqrt(tt) + 1)
    else:
        ks = 2

    # compute f(tt|0,1,w)
    p = 0.0
    if ks < kl:  # small t approximation
        K = int(ceil(ks))
        lower = -int(floor((K-1)/2))
        upper = int(ceil((K-1)/2))
        for k in range(lower, upper + 1):
            p += (w + 2 * k) * exp(-(pow(w + 2 * k, 2)) / 2 / tt)
        p /= sqrt(2 * pi * pow(tt, 3))
    else:  # large t approximation
        K = int(ceil(kl))
        for k in range(1, K + 1):
            p += k * exp(-(k * k) * (pi * pi) * tt / 2) * sin(k * pi * w)
        p *= pi

    return p

@njit(float64(float64, float64, float64, float64, float64, float64),
      cache=True)
def ddm_pdf_core(rt, a, z, v, ter, sv):
    """Core DDM PDF calculation including drift rate variability.

    Parameters use diffusion coefficient s=1:
        rt: response time (seconds)
        a:  boundary separation (~0.65-2.4)
        z:  relative starting point (0 to 1, e.g. 0.5 for unbiased)
        v:  drift rate (~0.1-0.5)
        ter: non-decision time (seconds)
        sv: drift rate variability (Gaussian SD)
    """
    if rt <= 0 or rt <= ter or a <= 0:
        return 0.0

    tt = (rt - ter) / (a * a)
    err = 1e-4
    p = ftt_01w(tt, z, err)

    if sv <= 1e-10:
        return p * exp(-v * a * z - (v * v * (rt - ter)) / 2.) / (a * a)
    else:
        sv_squared_rt = sv * sv * (rt - ter)

        if sv_squared_rt > 100:
            return 0.0

        exp_term = ((a * z * sv)**2 - 2 * a * v * z - (v * v) * (rt - ter)) / (2 * (sv_squared_rt + 1))
        scaling = 1 / sqrt(sv_squared_rt + 1)
        if exp_term < -500 or exp_term > 500:
            return 0.0

        return p * exp(exp_term) * scaling / (a * a)
