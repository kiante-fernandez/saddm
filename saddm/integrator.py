import numpy as np
from numba import njit
from scipy.special import roots_legendre
from .core import ddm_pdf_core

class DDMIntegrator:
    def __init__(self, n_points=15):
        self.n_points = n_points
        self.sz_nodes, self.sz_weights = roots_legendre(n_points)
        self.st_nodes, self.st_weights = roots_legendre(n_points)
        self.sa_nodes, self.sa_weights = roots_legendre(n_points)

    @staticmethod
    @njit
    def integrate(rt, a, z, v, ter, sv, sz, st, sa, sz_nodes, sz_weights, st_nodes, st_weights, sa_nodes, sa_weights):
        """One triple loop over the sz/st/sa grids. An inactive axis collapses to a
        single node with weight 1 and its validity guard is skipped."""
        # Threshold for when to consider a variability parameter active
        eps = 1e-6

        # Parameter validation - check for invalid parameters
        if (sz > 0 and sz > 2 * min(z, 1 - z)) or \
           (st > 0 and st > 2 * ter) or \
           (sa > 0 and sa > 2 * a):
            return 0.0

        # Check response time validity
        if rt <= 0 or rt <= ter:
            return 0.0

        # Determine which variability parameters are active
        sz_active = sz >= eps
        st_active = st >= eps
        sa_active = sa >= eps

        # Base case: No variability in sz, st, or sa
        if not sz_active and not st_active and not sa_active:
            return ddm_pdf_core(rt, a, z, v, ter, sv)

        # Build the per-axis grid: quadrature nodes/weights when active,
        # otherwise a single node at the fixed center with weight 1.
        if sz_active:
            z_grid = z + sz_nodes * (sz / 2)
            z_wts = sz_weights * 0.5
        else:
            z_grid = np.array([z], dtype=np.float64)
            z_wts = np.array([1.0], dtype=np.float64)

        if st_active:
            ter_grid = ter + st_nodes * (st / 2)
            ter_wts = st_weights * 0.5
        else:
            ter_grid = np.array([ter], dtype=np.float64)
            ter_wts = np.array([1.0], dtype=np.float64)

        if sa_active:
            a_grid = a + sa_nodes * (sa / 2)
            a_wts = sa_weights * 0.5
        else:
            a_grid = np.array([a], dtype=np.float64)
            a_wts = np.array([1.0], dtype=np.float64)

        total = 0.0
        for i in range(len(z_grid)):
            z_val = z_grid[i]
            if sz_active and (z_val < 0 or z_val > 1):
                continue  # Skip invalid z values

            for j in range(len(ter_grid)):
                ter_val = ter_grid[j]
                if st_active and (ter_val < 0 or ter_val >= rt):
                    continue  # Skip invalid ter values

                for k in range(len(a_grid)):
                    a_val = a_grid[k]
                    if sa_active and a_val <= 0.01:
                        continue  # Skip invalid a values

                    # Add weighted contribution
                    wt = z_wts[i] * ter_wts[j] * a_wts[k]
                    pdf = ddm_pdf_core(rt, a_val, z_val, v, ter_val, sv)
                    total += wt * pdf

        return total