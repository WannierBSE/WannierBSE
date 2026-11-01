"""
generate_vc_band

PURPOSE:
    Generates a list of (valence, conduction) band pairs using a vectorized

DESCRIPTION:
    approach. Python equivalent of generate_vc_band.m.

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

def generate_vc_band_py(Nv, Nc):
    """
    generate_vc_band_py
    
    PURPOSE:
        Generates an (Nv*Nc) x 2 array listing all unique pairs of
    
    DESCRIPTION:
        (valence, conduction) band indices using np.mgrid.
    
    INPUT ARGUMENTS:
        Nv (int): Number of valence bands.
        Nc (int): Number of conduction bands.
    
    OUTPUT:
        np.ndarray: A 2D NumPy array of shape (Nv*Nc, 2), where
        each row is [valence_index, conduction_index].
        Indices are 1-based, matching the MATLAB logic.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    try:
        # Ensure Nv and Nc are integers
        Nv = int(Nv)
        Nc = int(Nc)

        if Nv <= 0 or Nc <= 0:
            raise ValueError("Nv and Nc must be positive integers.")

        # --- Vectorized approach using np.mgrid ---
        # Create grids of indices: vi ranges from 1 to Nv, ci from 1 to Nc
        vi, ci = np.mgrid[1:Nv+1, 1:Nc+1]

        # Flatten the grids and stack them column-wise, then transpose
        # vi.ravel() creates a 1D array of valence indices (e.g., [1, 1, 2, 2])
        # ci.ravel() creates a 1D array of conduction indices (e.g., [1, 2, 1, 2])
        # np.vstack creates a 2x(Nv*Nc) array
        # .T transposes it to (Nv*Nc)x2
        vc_band_BSE = np.vstack((vi.ravel(), ci.ravel())).T
        # --- End vectorized approach ---

        return vc_band_BSE

    except Exception as e:
        print(f"Error in generate_vc_band_py: {e}", file=sys.stderr)
        sys.exit(1)

# Example Usage (optional)
if __name__ == '__main__':
    Nv_test = 2
    Nc_test = 2

    print(f"Testing generate_vc_band_py with Nv={Nv_test}, Nc={Nc_test}...")
    vc_pairs = generate_vc_band_py(Nv_test, Nc_test)

    print("Generated pairs:")
    print(vc_pairs)

    # Expected output:
    # [[1 1]
    #  [1 2]
    #  [2 1]
    #  [2 2]]

    expected = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    assert np.array_equal(vc_pairs, expected), "Test failed: Output does not match expected."
    print("Test passed.")
