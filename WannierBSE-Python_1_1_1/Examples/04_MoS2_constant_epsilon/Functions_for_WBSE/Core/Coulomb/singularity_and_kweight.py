"""
singularity_and_kweight

PURPOSE:
    Calculates the Coulomb singularity terms (Wq0) and k-point weights (wi)

DESCRIPTION:
    by calling the coarse and dense grid integration functions.
    Python equivalent of Singularity_and_kweight.m.

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

# Import the coarse and dense integration functions from the same directory
try:
    # Use relative import for package structure
    from .coulomb_integral_D3h import coulomb_integral_D3h_py
    from .coulomb_integral_D3h_dense import coulomb_integral_D3h_dense_py
except ImportError:
    # Fallback for direct execution/testing
    try:
        from coulomb_integral_D3h import coulomb_integral_D3h_py
        from coulomb_integral_D3h_dense import coulomb_integral_D3h_dense_py
    except ImportError:
        print("Error: Cannot import Coulomb integral functions.")
        print("Ensure 'coulomb_integral_D3h.py' and 'coulomb_integral_D3h_dense.py' are in the same directory.")
        sys.exit(1)

def singularity_and_kweight_py(N_de, N_co, Nv_coarse, Nv_dense, d, q_data, eps_q_2d_data, a, f):
    """
    singularity_and_kweight_py
    
    PURPOSE:
        Calculates the k-point-dependent singularity Wq0 and weights wi.
    
    DESCRIPTION:
        This function calls the coarse and dense grid integrators and combines
        their results into single arrays based on the k-point distribution.
    
    INPUT ARGUMENTS:
        N_de (int): Number of dense k-points.
        N_co (int): Number of coarse k-points.
        Nv_coarse (int): Number of k-points along coarse grid axis.
        Nv_dense (int): Number of k-points along dense grid axis.
        d (float): Effective layer thickness.
        q_data (np.ndarray): 1D array of q-points from the data file.
        eps_q_2d_data (np.ndarray): 1D array of corresponding epsilon values.
        a (float): Lattice constant.
        f (float): Scaling fraction for the dense grid.
    
    OUTPUT:
        tuple: Contains the following elements:
        Wq0 (np.ndarray): 1D array (N_kpt,) of singularity values.
        wi (np.ndarray): 1D array (N_kpt,) of k-point area weights.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    try:
        # 1. Calculate for the coarse grid
        # Vkkp -> Wq0_co, delta_kV -> wi_co
        Wq0_co, wi_co = coulomb_integral_D3h_py(
            Nv_coarse, a, d, q_data, eps_q_2d_data
        )

        # 2. Calculate for the dense grid
        # Wq0_de, wi_de = coulomb_integral_D3h_dense_py(
        Wq0_de, wi_de = coulomb_integral_D3h_dense_py(
            Nv_dense, a, d, q_data, eps_q_2d_data, f
        )

    except Exception as e:
        print(f"Error during Coulomb integral calculation: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Combine results based on k-point counts (N_de, N_co)
    # The k-mesh is structured with all dense points first, then coarse points.

    # Create an array for Wq0 with N_de copies of Wq0_de,
    # followed by N_co copies of Wq0_co.
    Wq0 = np.concatenate([
        np.full(N_de, Wq0_de),
        np.full(N_co, Wq0_co)
    ])

    # Create an array for wi with N_de copies of wi_de,
    # followed by N_co copies of wi_co.
    wi = np.concatenate([
        np.full(N_de, wi_de),
        np.full(N_co, wi_co)
    ])

    return Wq0, wi

# Example Usage (optional)
if __name__ == '__main__':
    # This block requires valid input data to run successfully
    print("Testing singularity_and_kweight_py...")

    # Create mock input data for testing
    _N_de = 10
    _N_co = 5
    _Nv_coarse = 2
    _Nv_dense = 3
    _d = 20.0
    _a = 3.18
    _f = 2.0

    # Mock dielectric data (must be sorted by q)
    _q_data = np.linspace(0, 1.0, 10)
    # Simple mock function: eps = 1 + q**2 + 0.1j * q
    _eps_data = 1.0 / (1.0 - _q_data**2 + 0.1j * _q_data).real + _q_data * 5
    _eps_data[0] = 1.0 # Handle q=0

    print(f"Mock params: N_de={_N_de}, N_co={_N_co}, Nv_coarse={_Nv_coarse}, Nv_dense={_Nv_dense}")

    try:
        Wq0_test, wi_test = singularity_and_kweight_py(
            _N_de, _N_co, _Nv_coarse, _Nv_dense, _d, _q_data, _eps_data, _a, _f
        )

        print("Test calculation successful.")
        print(f"  Wq0 array shape: {Wq0_test.shape}")
        print(f"  wi array shape: {wi_test.shape}")
        print(f"  Total k-points: {_N_de + _N_co}")

        # Check that the first N_de elements are the same (dense value)
        # and the last N_co elements are the same (coarse value)
        assert len(np.unique(Wq0_test[:_N_de])) == 1, "Dense Wq0 values are not uniform."
        assert len(np.unique(Wq0_test[_N_de:])) == 1, "Coarse Wq0 values are not uniform."
        assert len(np.unique(wi_test[:_N_de])) == 1, "Dense wi values are not uniform."
        assert len(np.unique(wi_test[_N_de:])) == 1, "Coarse wi values are not uniform."

        print(f"  Dense Wq0 value: {Wq0_test[0]:.4e}")
        print(f"  Coarse Wq0 value: {Wq0_test[-1]:.4e}")
        print(f"  Dense wi value: {wi_test[0]:.4e}")
        print(f"  Coarse wi value: {wi_test[-1]:.4e}")
        print("Test passed.")

    except Exception as e:
        print(f"Test failed: {e}", file=sys.stderr)
