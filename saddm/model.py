import numpy as np
from .integrator import DDMIntegrator
from .core import ddm_pdf_core
from math import log, isinf, isnan

class DDMModel:
    def __init__(self, n_points=15):
        """Initialize DDM model with integrator."""
        self.integrator = DDMIntegrator(n_points)
        self.min_p = 1e-10
        
    def _validate_parameters(self, a, z, v, ter, sv, sz, st, sa):
        """Validate DDM parameters and return boolean indicating validity."""
        if any(map(lambda x: isinf(x) or isnan(x), [a, z, v, ter, sv, sz, st, sa])):
            return False, "Parameter contains inf or nan values"
            
        if a <= 0.01:
            return False, f"Boundary separation (a={a}) must be > 0.01"
        if z < 0 or z > 1:
            return False, f"Starting point (z={z}) must be between 0 and 1"
        if ter < 0:
            return False, f"Non-decision time (ter={ter}) must be non-negative"
    
        # Variability parameters validation
        if sv < 0:
            return False, f"Drift variability (sv={sv}) must be non-negative"
        if sz < 0:
            return False, f"Starting point variability (sz={sz}) must be non-negative"
        if st < 0:
            return False, f"Non-decision time variability (st={st}) must be non-negative"
        if sa < 0:
            return False, f"Boundary variability (sa={sa}) must be non-negative"
        if sa > 2 * a:
            return False, f"Boundary variability (sa={sa}) must be <= 2*a (a={a})"

        if ter - st/2 < 0:
            return False, f"Non-decision time variability (st={st}) too large for ter={ter}"
            
        # Starting point variability boundary checks
        if z + sz/2 > 1 or z - sz/2 < 0:
            return False, f"Starting point variability (sz={sz}) causes z to exceed limits at z={z}"
            
        return True, "Parameters valid"
        
    def pdf(self, rt, a, z, v, ter, sv=0.0, sz=0.0, st=0.0, sa=0.0, strict=False, validate=True):
        """Compute full DDM PDF with all variability parameters and validation."""
        # Parameter validation
        if validate:
            valid, reason = self._validate_parameters(a, z, v, ter, sv, sz, st, sa)
            if not valid:
                if strict:
                    raise ValueError(f"Invalid parameters: {reason}")
                else:
                    return self.min_p
        if rt <= ter:
            return self.min_p

        # Clean small variability parameters
        sv = 0.0 if sv < 1e-6 else sv
        sz = 0.0 if sz < 1e-6 else sz
        st = 0.0 if st < 1e-6 else st
        sa = 0.0 if sa < 1e-6 else sa
        
        # Handle base case with only sv or no variability
        if sz == 0 and st == 0 and sa == 0:
            try:
                p = ddm_pdf_core(rt, a, z, v, ter, sv)
            except (OverflowError, ValueError):
                return self.min_p
        else:
            try:
                p = self.integrator.integrate(rt, a, z, v, ter, sv, sz, st, sa,
                                            self.integrator.sz_nodes,
                                            self.integrator.sz_weights,
                                            self.integrator.st_nodes,
                                            self.integrator.st_weights,
                                            self.integrator.sa_nodes,
                                            self.integrator.sa_weights)
            except (OverflowError, ValueError):
                return self.min_p
        
        return max(p, self.min_p)
    
    def log_likelihood(self, params, data, strict=False, validate=True):
        """Calculate log-likelihood of data given parameters with validation."""
        a, z, v, ter, sv, sz, st, sa = params
        
        if validate:
            valid, reason = self._validate_parameters(a, z, v, ter, sv, sz, st, sa)
            if not valid:
                if strict:
                    raise ValueError(f"Invalid parameters: {reason}")
                return -np.inf  # Return negative infinity for invalid parameters
        
        total = 0.0 
        data = np.asarray(data)
        
        for rt, choice in data:
            try:
                eff_v = -v if choice > 0 else v
                eff_z = 1 - z if choice > 0 else z
                
                # Calculate PDF for this trial
                p = self.pdf(abs(rt), a, eff_z, eff_v, ter, sv, sz, st, sa)
                total += log(p)
                
            except (ValueError, OverflowError):
                return -np.inf
                
        if isinf(total) or isnan(total):
            return -np.inf
            
        return total  # Return log-likelihood