"""
folding_Gvector

PURPOSE:
    Generates reciprocal-lattice G-vectors using the direct cutoff.

DESCRIPTION:
    The omitted cutoff defaults to 1, based on the lattice vectors b1 and b2.
    Python equivalent of Folding_Gvector.m.

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

def folding_Gvector_py(b1, b2, G_dir_max=None):
    """
    folding_Gvector_py
    
    PURPOSE:
        Generates cutoff-based reciprocal-lattice vectors.
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        b1 (np.ndarray): Reciprocal lattice vector 1, shape (2,).
        b2 (np.ndarray): Reciprocal lattice vector 2, shape (2,).
    
    OUTPUT:
        np.ndarray: A (9, 2) array containing the G-vectors.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    try:
        # Ensure inputs are flat 1D arrays of shape (2,)
        b1_flat = np.asarray(b1).flatten()
        b2_flat = np.asarray(b2).flatten()

        if b1_flat.shape != (2,) or b2_flat.shape != (2,):
            raise ValueError("Input vectors b1 and b2 must be 2D.")

        cutoff = (
            1.0
            if G_dir_max is None or (isinstance(G_dir_max, str) and not G_dir_max.strip())
            else float(G_dir_max)
        )
        if not np.isfinite(cutoff) or cutoff < 0:
            raise ValueError("G_dir_max must be finite and nonnegative")
        if cutoff == 0:
            return np.zeros((1, 2), dtype=float)
        limit = int(np.ceil(cutoff))
        cart_cutoff = 1.005 * cutoff * np.linalg.norm(b1_flat)
        vectors = []
        for n in range(-limit, limit + 1):
            for m in range(-limit, limit + 1):
                g = m * b1_flat + n * b2_flat
                if np.linalg.norm(g) <= cart_cutoff:
                    vectors.append(g)
        return np.asarray(vectors, dtype=float)

    except Exception as e:
        print(f"Error in folding_Gvector_py: {e}", file=sys.stderr)
        raise # Re-raise the exception to be caught by the caller
