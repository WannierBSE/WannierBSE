"""
screened_coulomb_interaction

PURPOSE:
    Calculates the screened Coulomb interaction V(q)/epsilon(q).

DESCRIPTION:
    Python equivalent of Screened_Coulomb_interaction.m.

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
import traceback

# Import translated dependencies
try:
    # Use relative import for package structure
    from Functions_for_WBSE.Core.Direct.wave_vector_folding import wave_vector_folding_py
    from .epsilon_q2d import epsilon_q2d_py
    from .bare_coulomb_q2d import bare_coulomb_q2d_py
except ImportError:
    # Fallback for direct execution/testing
    try:
        from wave_vector_folding import wave_vector_folding_py
        from epsilon_q2d import epsilon_q2d_py
        from bare_coulomb_q2d import bare_coulomb_q2d_py
    except ImportError:
        print("Error: Cannot import dependencies 'wave_vector_folding_py' or 'epsilon_q2d_py'.", file=sys.stderr)
        sys.exit(1)

def screened_coulomb_interaction_py(
    q,
    b1,
    b2,
    d,
    q_data,
    eps_q_2d_data,
    G_dir_max=None,
    verbose=False,
):
    """
    screened_coulomb_interaction_py
    
    PURPOSE:
        Calculates the bare Coulomb potential V(q) and dielectric screening
    
    DESCRIPTION:
        epsilon(q).
    
    INPUT ARGUMENTS:
        q (np.ndarray): Nx2 array of q-vectors (k - k').
        b1 (np.ndarray): Reciprocal lattice vector 1 (2 elements).
        b2 (np.ndarray): Reciprocal lattice vector 2 (2 elements).
        d (float): Effective layer thickness.
        q_data (np.ndarray): 1D array of q-points from the data file.
        eps_q_2d_data (np.ndarray): 1D array of corresponding epsilon
        values.
        verbose (bool, optional): If True, print debug info. Defaults to
        False.
    
    OUTPUT:
        tuple: (Vq, epsilon_q)
        Vq (np.ndarray): 1D array (N,) of the bare Coulomb potential
        V(q).
        epsilon_q (np.ndarray): 1D array (N,) of the interpolated
        epsilon(q).
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    if verbose:
        print(f"  [Debug] screened_coulomb_interaction_py: Folding {q.shape[0]} q-vectors...")

    try:
        # 1. Fold q-vectors to 1st BZ
        # q_mag and MinNdkG are 1D arrays of shape (N,)
        q_mag, MinNdkG = wave_vector_folding_py(q, b1, b2, G_dir_max)

        # 2. Get dielectric screening
        if verbose:
            print("  [Debug] screened_coulomb_interaction_py: Interpolating epsilon(q)...")
        epsilon_q = epsilon_q2d_py(q_data, eps_q_2d_data, q_mag)

        # 3. Calculate V(q)
        # This is the formula from the MATLAB script:
        # Vq = (1-((1-exp(-(d.*q_mag)))./(d.*q_mag))).*((4*pi)./(d.*(q_mag.^2)));
        
        # Suppress "divide by zero" and "invalid value" warnings for q=0
        Vq = bare_coulomb_q2d_py(q_mag, d)

        if verbose:
            print("  [Debug] screened_coulomb_interaction_py: Vq and epsilon_q calculated.")

        return Vq, epsilon_q

    except Exception as e:
        print(f"--- ERROR in screened_coulomb_interaction_py ---", file=sys.stderr)
        print(f"Input shapes: q={q.shape}, b1={b1.shape}, b2={b2.shape}", file=sys.stderr)
        print(f"Exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
