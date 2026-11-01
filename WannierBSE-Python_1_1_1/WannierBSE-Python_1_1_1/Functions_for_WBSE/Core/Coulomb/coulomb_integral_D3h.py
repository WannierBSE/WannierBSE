"""
coulomb_integral_D3h

PURPOSE:
    Calculates the 2D Coulomb integral for the coarse k-mesh (Nt).

DESCRIPTION:
    Python equivalent of CoulombIntegral_D3h.m.

    This function performs numerical 2D integration (dblquad) over a
    hexagonal Brillouin zone by splitting it into rectangular and
    triangular sections.

INPUT ARGUMENTS:
    See the annotated signature and detailed description.

OUTPUT:
    See the annotated signature and detailed description.

PHYSICS/NUMERICAL NOTES:
    Preserves the numerical conventions, units, indexing, and array ordering established by the caller.

SIDE EFFECTS:
    None unless explicitly documented by this function.
"""
import sys
import numpy as np
from scipy.integrate import dblquad

# Import the epsilon interpolator from the same directory
try:
    # Use relative import for package structure
    from .epsilon_q2d import epsilon_q2d_py
    from .bare_coulomb_q2d import bare_coulomb_q2d_py
except ImportError:
    # Fallback for direct execution/testing
    try:
        from epsilon_q2d import epsilon_q2d_py
        from bare_coulomb_q2d import bare_coulomb_q2d_py
    except ImportError:
        print("Error: Cannot import epsilon_q2d_py. Ensure epsilon_q2d.py is in the same directory.", file=sys.stderr)
        sys.exit(1)

def coulomb_integral_D3h_py(Nt, a, d, q_data, eps_q_2d_data):
    """
    coulomb_integral_D3h_py
    
    PURPOSE:
        Calculates the singularity (Vkkp) and area (delta_kV) for the coarse
    
    DESCRIPTION:
        grid.
    
    INPUT ARGUMENTS:
        Nt (int): Number of k-points along the coarse grid axis (Nv_coarse).
        a (float): Lattice constant.
        d (float): Effective layer thickness.
        q_data (np.ndarray): 1D array of q-points from the data file.
        eps_q_2d_data (np.ndarray): 1D array of corresponding epsilon values.
    
    OUTPUT:
        tuple: Contains the following elements:
        Vkkp (float): The numerically integrated Coulomb singularity.
        delta_kV (float): The area of one k-point parallelogram.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    # --- Define Integrand Function ---
    # Final screened Coulomb interaction W(q) = V(q) / epsilon(q)
    # This is the function we will integrate.
    def f(ky, kx): # Note: scipy.dblquad expects f(y, x)
        """
        f
        
        PURPOSE:
            F.
        
        DESCRIPTION:
            Implements the operation described by the function name and signature.
        
        INPUT ARGUMENTS:
            See the annotated signature and detailed description.
        
        OUTPUT:
            See the annotated signature and detailed description.
        
        PHYSICS/NUMERICAL NOTES:
            Array ordering, units, indexing, and numerical conventions follow
            the caller and the annotated function signature.
        
        SIDE EFFECTS:
            None unless explicitly documented by this function.
        """
        q_mag_sq = kx**2 + ky**2
        if q_mag_sq < 1e-12:
            return 0.0 # Integrand is 0 at q=0
        
        q_mag = np.sqrt(q_mag_sq)

        V_q = float(bare_coulomb_q2d_py(np.asarray([q_mag]), d)[0])

        # 2. Calculate epsilon(q) by calling the imported function
        eps_q = epsilon_q2d_py(q_data, eps_q_2d_data, q_mag)
        
        if eps_q < 1e-12: # Avoid division by zero
            return 0.0
            
        return V_q / eps_q

    # --- Define Integration Geometry (Hexagonal Unit Cell) ---
    # This geometry matches the MATLAB implementation
    v1 = (np.pi / a) * np.array([2.0/3.0, 2.0/np.sqrt(3.0)])
    v1_unit = v1 / float(Nt) # Convert Nt to float for division
    Nv1_unit = np.linalg.norm(v1_unit)

    h = Nv1_unit * (np.sqrt(3.0) / 2.0)
    l1 = Nv1_unit / np.sqrt(3.0)
    l2 = Nv1_unit / (2.0 * np.sqrt(3.0))

    # Integration tolerances (looser than MATLAB default to match script)
    AbsTol = 1e-7
    RelTol = 5e-4
    m = -np.sqrt(3.0) # Slope

    # --- Perform Numerical Integration (dblquad) ---
    # We split the hexagon into a central rectangle and 4 triangles
    # Note: dblquad is f(y, x), so limits are g(x) to h(x) for y,
    # and a to b for x.
    try:
        # --- 1. Central Rectangle ---
        # limits: x from -l2*sqrt(3)/2 to l2*sqrt(3)/2
        #         y from -l2 to l2/2
        rec_kxmin = -l2 * (np.sqrt(3.0) / 2.0)
        rec_kxmax = l2 * (np.sqrt(3.0) / 2.0)
        rec_kymin = -l2
        rec_kymax = l2 / 2.0

        Vkkp_rec, err_rec = dblquad(
            f, rec_kxmin, rec_kxmax, 
            lambda x: rec_kymin, lambda x: rec_kymax, 
            epsabs=AbsTol, epsrel=RelTol
        )

        # --- 2. Left Triangle ---
        LT_kxmin = -Nv1_unit * np.cos(np.pi / 3.0)
        LT_kxmax = -l2 * (np.sqrt(3.0) / 2.0)
        def LT_kymin(kx): return -l2
        def LT_kymax(kx): return l1 + m * np.abs(kx)

        Vkkp_LT, err_LT = dblquad(
            f, LT_kxmin, LT_kxmax, 
            LT_kymin, LT_kymax, 
            epsabs=1e-13, epsrel=1e-10
        )

        # --- 3. Right Triangle ---
        RT_kxmin = l2 * (np.sqrt(3.0) / 2.0)
        RT_kxmax = Nv1_unit * np.cos(np.pi / 3.0)
        def RT_kymin(kx): return -l2
        def RT_kymax(kx): return l1 + m * np.abs(kx)

        Vkkp_RT, err_RT = dblquad(
            f, RT_kxmin, RT_kxmax, 
            RT_kymin, RT_kymax, 
            epsabs=1e-13, epsrel=1e-10
        )

        # --- 4. Top Triangle ---
        TT_kxmin = -l2 * (np.sqrt(3.0) / 2.0)
        TT_kxmax = l2 * (np.sqrt(3.0) / 2.0)
        def TT_kymin(kx): return (l2 / 2.0)
        def TT_kymax(kx): return l1 + m * np.abs(kx)

        Vkkp_TT, err_TT = dblquad(
            f, TT_kxmin, TT_kxmax, 
            TT_kymin, TT_kymax, 
            epsabs=1e-13, epsrel=1e-10
        )

    except Exception as e:
        print(f"Error during numerical integration in coulomb_integral_D3h_py: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Sum results and apply final scaling ---
    Vkkp_total = Vkkp_rec + Vkkp_LT + Vkkp_RT + Vkkp_TT

    # Apply 1/d scaling as in MATLAB code
    Vkkp = Vkkp_total

    # Area of one unit parallelogram
    delta_kV = h * Nv1_unit / 2.0

    return Vkkp, delta_kV
