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
        """
        Adaptive integration method that selects the optimal integration routine 
        based on which variability parameters are active.
        """
        # Threshold for when to consider a variability parameter active
        eps = 1e-6
        
        # Parameter validation - check for invalid parameters
        if (sz > 0 and sz > 2 * min(z, 1 - z)) or \
           (st > 0 and st > 2 * ter) or \
           (sa > 0 and sa > a):
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
        
        # Prepare total accumulator
        total = 0.0
        n_points = len(sz_nodes)
        
        # Case 1: Only sz is active (1D integration)
        if sz_active and not st_active and not sa_active:
            # Scale nodes and weights for z
            sz_scaled = z + sz_nodes * (sz / 2)
            sz_weights_scaled = sz_weights * 0.5
            
            for i in range(n_points):
                z_val = sz_scaled[i]
                if z_val < 0 or z_val > 1:
                    continue  # Skip invalid z values
                    
                # Add weighted contribution
                pdf = ddm_pdf_core(rt, a, z_val, v, ter, sv)
                total += sz_weights_scaled[i] * pdf
                
            return total
        
        # Case 2: Only st is active (1D integration)
        if not sz_active and st_active and not sa_active:
            # Scale nodes and weights for ter
            st_scaled = ter + st_nodes * (st / 2)
            st_weights_scaled = st_weights * 0.5
            
            for j in range(n_points):
                ter_val = st_scaled[j]
                if ter_val < 0 or ter_val >= rt:
                    continue  # Skip invalid ter values
                    
                # Add weighted contribution
                pdf = ddm_pdf_core(rt, a, z, v, ter_val, sv)
                total += st_weights_scaled[j] * pdf
                
            return total
        
        # Case 3: Only sa is active (1D integration)
        if not sz_active and not st_active and sa_active:
            # Scale nodes and weights for a
            sa_scaled = a + sa_nodes * (sa / 2)
            sa_weights_scaled = sa_weights * 0.5
            
            for k in range(n_points):
                a_val = sa_scaled[k]
                if a_val <= 0.01:
                    continue  # Skip invalid a values
                    
                # Add weighted contribution
                pdf = ddm_pdf_core(rt, a_val, z, v, ter, sv)
                total += sa_weights_scaled[k] * pdf
                
            return total
        
        # Case 4: sz and st are active (2D integration)
        if sz_active and st_active and not sa_active:
            # Scale nodes and weights
            sz_scaled = z + sz_nodes * (sz / 2)
            sz_weights_scaled = sz_weights * 0.5
            st_scaled = ter + st_nodes * (st / 2)
            st_weights_scaled = st_weights * 0.5
            
            for i in range(n_points):
                z_val = sz_scaled[i]
                if z_val < 0 or z_val > 1:
                    continue
                    
                for j in range(n_points):
                    ter_val = st_scaled[j]
                    if ter_val < 0 or ter_val >= rt:
                        continue
                        
                    # Calculate weighted contribution
                    wt = sz_weights_scaled[i] * st_weights_scaled[j]
                    pdf = ddm_pdf_core(rt, a, z_val, v, ter_val, sv)
                    total += wt * pdf
                    
            return total
        
        # Case 5: sz and sa are active (2D integration)
        if sz_active and not st_active and sa_active:
            # Scale nodes and weights
            sz_scaled = z + sz_nodes * (sz / 2)
            sz_weights_scaled = sz_weights * 0.5
            sa_scaled = a + sa_nodes * (sa / 2)
            sa_weights_scaled = sa_weights * 0.5
            
            for i in range(n_points):
                z_val = sz_scaled[i]
                if z_val < 0 or z_val > 1:
                    continue
                    
                for k in range(n_points):
                    a_val = sa_scaled[k]
                    if a_val <= 0.01:
                        continue
                        
                    # Calculate weighted contribution
                    wt = sz_weights_scaled[i] * sa_weights_scaled[k]
                    pdf = ddm_pdf_core(rt, a_val, z_val, v, ter, sv)
                    total += wt * pdf
                    
            return total
        
        # Case 6: st and sa are active (2D integration)
        if not sz_active and st_active and sa_active:
            # Scale nodes and weights
            st_scaled = ter + st_nodes * (st / 2)
            st_weights_scaled = st_weights * 0.5
            sa_scaled = a + sa_nodes * (sa / 2)
            sa_weights_scaled = sa_weights * 0.5
            
            for j in range(n_points):
                ter_val = st_scaled[j]
                if ter_val < 0 or ter_val >= rt:
                    continue
                    
                for k in range(n_points):
                    a_val = sa_scaled[k]
                    if a_val <= 0.01:
                        continue
                        
                    # Calculate weighted contribution
                    wt = st_weights_scaled[j] * sa_weights_scaled[k]
                    pdf = ddm_pdf_core(rt, a_val, z, v, ter_val, sv)
                    total += wt * pdf
                    
            return total
        
        # Case 7: All variability parameters are active (3D integration)
        # Scale nodes and weights
        sz_scaled = z + sz_nodes * (sz / 2)
        sz_weights_scaled = sz_weights * 0.5
        st_scaled = ter + st_nodes * (st / 2)
        st_weights_scaled = st_weights * 0.5
        sa_scaled = a + sa_nodes * (sa / 2)
        sa_weights_scaled = sa_weights * 0.5
        
        for i in range(n_points):
            z_val = sz_scaled[i]
            if z_val < 0 or z_val > 1:
                continue
                
            for j in range(n_points):
                ter_val = st_scaled[j]
                if ter_val < 0 or ter_val >= rt:
                    continue
                    
                for k in range(n_points):
                    a_val = sa_scaled[k]
                    if a_val <= 0.01:
                        continue
                        
                    # Calculate weighted contribution
                    wt = sz_weights_scaled[i] * st_weights_scaled[j] * sa_weights_scaled[k]
                    pdf = ddm_pdf_core(rt, a_val, z_val, v, ter_val, sv)
                    total += wt * pdf
                    
        return total