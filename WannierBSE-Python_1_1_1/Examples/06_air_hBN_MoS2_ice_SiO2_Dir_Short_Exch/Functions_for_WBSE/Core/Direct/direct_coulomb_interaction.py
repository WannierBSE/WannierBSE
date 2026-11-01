"""
direct_coulomb_interaction

PURPOSE:
    Calculates one column (for a k-point 'j') of the direct Coulomb

DESCRIPTION:
    interaction matrix, Vd.

    Python equivalent of Direct_Coulomb_interaction.m.

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
    from .wave_vector_folding import wave_vector_folding_py
    from .folding_Gvector import folding_Gvector_py
    from .phase_tau_in_Vd import phase_tau_in_Vd_py
    from .Vd_singularity import Vd_singularity_py
except ImportError:
    # Fallback for direct execution/testing
    try:
        from wave_vector_folding import wave_vector_folding_py
        from folding_Gvector import folding_Gvector_py
        from phase_tau_in_Vd import phase_tau_in_Vd_py
        from Vd_singularity import Vd_singularity_py
    except ImportError:
        print("Error: Cannot import dependencies in 'direct_coulomb_interaction_py'.", file=sys.stderr)
        sys.exit(1)

# --- DEBUG FLAG ---
DEBUG_PRINT = False
# -------------------

def direct_coulomb_interaction_py(
    Vq, epslnq, k, b1, b2, Wq0_j, wi_j, q,
    tau, Basis_tau_index_cell, WF_center, N_atoms,
    Cc_kex, Cv, c, cc, v, vv, j, e, eps0, verbose=False, G_dir_max=None
):
    """
    direct_coulomb_interaction_py
    
    PURPOSE:
        Calculates the Vd vector for a single k-point 'j'.
    
    DESCRIPTION:
        This implementation replicates the MATLAB Vd calculation, including the
        asymmetric conjugation required for the final Vd = Mc * conj(Mv) step.
    
    INPUT ARGUMENTS:
        (all args): Pre-calculated or loaded by BSE.py and hbse_worker.py.
    
    OUTPUT:
        np.ndarray: The Vd vector for a single k-point 'j'.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    try:
        N_kpt = k.shape[0]
        NB = Cc_kex.shape[1]

        # --- Step 1: Fold q-vectors to get the G-vector indices ---
        _, MinNdkG = wave_vector_folding_py(q, b1, b2, G_dir_max)
        Gi = folding_Gvector_py(b1, b2, G_dir_max)
        if DEBUG_PRINT:
            print(f"  [Py Vd j={j}] Step 1: MinNdkG mean: {np.mean(MinNdkG):.18e}")

        # --- Step 2: Get the pre-calculated phase-modified coefficients ---
        Cc_kex_tilde, Cv_tilde = phase_tau_in_Vd_py(k, Cc_kex, Cv, tau, WF_center, verbose=verbose)
        
        if DEBUG_PRINT:
            print(f"  [Py Vd j={j}] Step 2: Cc_tilde mean (real): {np.mean(np.real(Cc_kex_tilde)):.18e}")
            print(f"  [Py Vd j={j}] Step 2: Cv_tilde mean (real): {np.mean(np.real(Cv_tilde)):.18e}")

        # --- Step 3: Calculate the exp(i*G*tau) phase factor ---
        Gkkp = Gi[MinNdkG - 1, :] # Shape (N_kpt, 2)
        Gkkp_3d = np.hstack([Gkkp, np.zeros((N_kpt, 1))]) # Add z=0 for dot product
        expiGtau = np.zeros((N_kpt, NB), dtype=complex)

        for Ni in range(N_atoms):
            current_basis_indices_1based = Basis_tau_index_cell[Ni]
            if not current_basis_indices_1based:
                continue

            current_basis_indices_0based = np.array(current_basis_indices_1based) - 1
            tau_Ni = tau[Ni, :][np.newaxis, :] # Shape (1, 3)
            G_dot_tau = np.sum(Gkkp_3d * tau_Ni, axis=1) # Shape (N_kpt,)
            phase_vec = np.exp(1j * G_dot_tau) # Shape (N_kpt,)
            expiGtau[:, current_basis_indices_0based] = phase_vec[:, np.newaxis]

        if DEBUG_PRINT:
            print(f"  [Py Vd j={j}] Step 3: expiGtau mean (real): {np.mean(np.real(expiGtau)):.18e}")

        # --- Step 4: Calculate Matrix Elements Mc and Mv (The core overlap) ---
        Cj_c_vec = Cc_kex_tilde[j, :, cc-1][np.newaxis, :] # Shape (1, NB)
        Cj_v_vec = Cv_tilde[j, :, vv-1][np.newaxis, :] # Shape (1, NB)

        Mc = np.sum(
            np.conj(Cc_kex_tilde[:, :, c-1]) * expiGtau * Cj_c_vec,
            axis=1 # Sum over NB
        )

        Mv = np.sum(
            np.conj(Cv_tilde[:, :, v-1]) * expiGtau * Cj_v_vec,
            axis=1 # Sum over NB
        )

        if DEBUG_PRINT:
            print(f"  [Py Vd j={j}] Step 4: Mc mean: {np.mean(Mc):.18e}, Mv mean: {np.mean(Mv):.18e}")

        # --- Step 5: Calculate Vd vector (Final result uses CONJUGATE of Mv) ---
        Vd = -((e**2) / (4 * np.pi * eps0)) * (Vq / epslnq) * Mc * np.conj(Mv)

        if DEBUG_PRINT:
            print(f"  [Py Vd j={j}] Step 5: Vd (off-diag) mean: {np.mean(Vd):.18e}")

        # --- Step 6: Handle the diagonal (k=k') singularity ---
        Vd[j] = Vd_singularity_py(
            e, eps0, Wq0_j, wi_j, Cc_kex_tilde, Cv_tilde, c, cc, v, vv, j,
            verbose=verbose
        )

        if DEBUG_PRINT:
            print(f"  [Py Vd j={j}] Step 6: Vd[j]: {Vd[j]:.18e}")
            print(f"  [Py Vd j={j}] Final Vd mean: {np.mean(Vd):.18e}")

        return Vd

    except Exception as e:
        print(f"--- ERROR in direct_coulomb_interaction_py ---", file=sys.stderr)
        print(f"Inputs: j={j}, c={c}, cc={cc}, v={v}, vv={vv}", file=sys.stderr)
        print(f"Exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
