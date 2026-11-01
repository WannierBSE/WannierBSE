"""
wave_vector_folding

PURPOSE:
    Calculates the folded q-vector magnitude and index of the G-vector

DESCRIPTION:
    that folds q into the 1st Brillouin Zone.
    Python equivalent of wave_vector_folding.m, with NumPy vectorization.

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
from pathlib import Path
import traceback # Import for detailed error logging

# Import the translated G-vector generator
try:
    # Use relative import for package structure
    from .folding_Gvector import folding_Gvector_py
except ImportError:
    # Fallback for direct execution/testing
    try:
        from folding_Gvector import folding_Gvector_py
    except ImportError:
        print("Error: Cannot import folding_Gvector_py.", file=sys.stderr)
        sys.exit(1)

def wave_vector_folding_py(q, b1, b2, G_dir_max=None):
    """
    wave_vector_folding_py
    
    PURPOSE:
        Finds the minimum magnitude of q + G by testing a set of G vectors.
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        q (array-like): Nx2 array of q-vectors (k - k').
        Can be MATLAB array.array or numpy.ndarray.
        b1 (array-like): Reciprocal lattice vector 1 (2 elements).
        b2 (array-like): Reciprocal lattice vector 2 (2 elements).
    
    OUTPUT:
        tuple: Contains the following elements:
        q_mag (np.ndarray): 1D array (N,) of the minimum |q + G|
        magnitudes.
        MinNdkG (np.ndarray): 1D array (N,) of the 1-based indices
        (relative to Gi) of the minimizing G-
        vector.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    try:
        # --- FIX: Ensure inputs are NumPy arrays ---
        # This handles data types passed from MATLAB (like array.array)
        q_np = np.asarray(q)
        b1_np = np.asarray(b1)
        b2_np = np.asarray(b2)
        # --- END FIX ---

        # Get the 9 G-vectors (shape (9, 2))
        Gi = folding_Gvector_py(b1_np, b2_np, G_dir_max)

        # Handle case where q is a single vector (1x2) from MATLAB
        if q_np.ndim == 1:
            q_np = q_np.reshape(1, -1)

        # Use broadcasting to create a (N_q, 9, 2) array
        # q_np[:, np.newaxis, :] -> (N_q, 1, 2)
        # Gi[np.newaxis, :, :]   -> (1, 9, 2)
        qG = q_np[:, np.newaxis, :] + Gi[np.newaxis, :, :]

        # Calculate norms along the last axis (axis=2)
        # NqG shape is (N_q, 9)
        NqG = np.linalg.norm(qG, axis=2)

        # Find the minimum magnitude along axis 1 (the G-vectors).
        q_mag = np.min(NqG, axis=1)

        # Preserve argmin for unique minima, then resolve only ties with
        # the validated inversion-symmetric local rule.
        MinNdkG_0based = np.argmin(NqG, axis=1)
        tie_tol_factor = 1e-12
        for qi in range(q_np.shape[0]):
            tie_tol = tie_tol_factor * max(1.0, float(q_mag[qi]))
            tied = np.flatnonzero(np.abs(NqG[qi, :] - q_mag[qi]) <= tie_tol)
            if tied.size > 1:
                dominant_component = int(np.argmax(np.abs(q_np[qi, :])))
                orientation = float(np.sign(q_np[qi, dominant_component]))
                if orientation == 0.0:
                    orientation = 1.0
                transformed_G = orientation * Gi[tied, :]
                selected_local = min(
                    range(tied.size),
                    key=lambda local: (
                        float(transformed_G[local, 0]),
                        float(transformed_G[local, 1]),
                    ),
                )
                MinNdkG_0based[qi] = tied[selected_local]

        # Return 1-based index to match MATLAB's logic
        MinNdkG = MinNdkG_0based + 1

        return q_mag, MinNdkG

    except Exception as e:
        print(f"Error in wave_vector_folding_py: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Re-raise the exception to be caught by MATLAB
        raise
