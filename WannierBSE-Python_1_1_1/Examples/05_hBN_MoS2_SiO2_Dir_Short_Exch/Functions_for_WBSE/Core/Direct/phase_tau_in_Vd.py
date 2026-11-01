"""
phase_tau_in_Vd

PURPOSE:
    Calculates the k-dependent phase factor exp(-i*k*tau) and

DESCRIPTION:
    absorbs it into the C (eigenvector) matrices.
    Python equivalent of phase_tau_in_Vd.m.

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

def phase_tau_in_Vd_py(k, Cc, Cv, tau, WF_center, verbose=False):
    """
    phase_tau_in_Vd_py
    
    PURPOSE:
        Calculates C_tilde(k) = C(k) * exp(-i*k*tau).
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        k (np.ndarray): K-points array, shape (N_kpt, 2).
        Cc (np.ndarray): Conduction band coefficients, shape (N_kpt, NB, Nc).
        Cv (np.ndarray): Valence band coefficients, shape (N_kpt, NB, Nv).
        tau (np.ndarray): Atomic positions, shape (N_atoms, 3).
        WF_center (np.ndarray): Wannier function centers, shape (NB, M).
        Assumes col 2 (index 1) is the 1-based atom index.
        verbose (bool, optional): If True, print debug info. Defaults to False.
    
    OUTPUT:
        tuple: (Cc_tilde, Cv_tilde)
        Cc_tilde (np.ndarray): Phased CB coefficients, shape (N_kpt, NB, Nc).
        Cv_tilde (np.ndarray): Phased VB coefficients, shape (N_kpt, NB, Nv).
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    if verbose:
        print("  [Debug] phase_tau_in_Vd_py: Calculating exp(-ik*tau)...")

    try:
        N_kpt = k.shape[0]
        NB = WF_center.shape[0]

        # 1. Create mapping from basis index (j) to atomic position (tau_I)
        # Get the 1-based atom index for each basis function
        basis_atom_indices = WF_center[:, 1].astype(int) - 1 # Now 0-based

        # Use advanced indexing to get the (x,y,z) position for each basis function
        # tau_j will have shape (NB, 3)
        tau_j = tau[basis_atom_indices, :]

        # 2. Add z=0 to k-points for dot product
        k_3d = np.hstack([k, np.zeros((N_kpt, 1))]) # Shape (N_kpt, 3)

        # 3. Calculate the k.tau dot product for all k and all j
        # (N_kpt, 3) @ (3, NB) -> (N_kpt, NB)
        k_dot_tau = k_3d @ tau_j.T

        # 4. Calculate the complex phase factor
        expiktau = np.exp(-1j * k_dot_tau) # Shape (N_kpt, NB)

        # 5. Apply the phase to all bands using broadcasting
        # expiktau[:, :, np.newaxis] -> (N_kpt, NB, 1)
        # This will multiply the (N_kpt, NB) phase with each band
        
        Cv_tilde = Cv * expiktau[:, :, np.newaxis]
        Cc_tilde = Cc * expiktau[:, :, np.newaxis]

        if verbose:
            print("  [Debug] phase_tau_in_Vd_py: C_tilde matrices calculated.")

        return Cc_tilde, Cv_tilde

    except Exception as e:
        print(f"--- ERROR in phase_tau_in_Vd_py ---", file=sys.stderr)
        print(f"Input shapes: k={k.shape}, Cc={Cc.shape}, Cv={Cv.shape}")
        print(f"Exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
