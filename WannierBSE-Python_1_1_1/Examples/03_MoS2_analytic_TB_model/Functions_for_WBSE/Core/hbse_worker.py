"""
hbse_worker

PURPOSE:
    Contains the `calculate_hbse_column` function, which performs the

DESCRIPTION:
    work for a single k-point 'j' in the HBSE matrix construction.
    This function is designed to be called in parallel by joblib.

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
import time

# Import translated dependencies
try:
    # Use relative import for package structure
    from Functions_for_WBSE.Core.Coulomb.epsilon_q2d import epsilon_q2d_py
    from Functions_for_WBSE.Core.Coulomb.screened_coulomb_interaction import screened_coulomb_interaction_py
    from Functions_for_WBSE.Core.Direct.Vd_singularity import Vd_singularity_py
    from Functions_for_WBSE.Core.Direct.direct_coulomb_interaction import direct_coulomb_interaction_py
    from Functions_for_WBSE.Core.Direct.folding_Gvector import folding_Gvector_py
    from Functions_for_WBSE.Core.Direct.wave_vector_folding import wave_vector_folding_py
    from Functions_for_WBSE.Core.Coulomb.bare_coulomb_q2d import bare_coulomb_q2d_py
    from Functions_for_WBSE.Core.Exchange.exchange_coulomb_interaction import (
        exchange_coulomb_interaction_batched_j_py,
        exchange_coulomb_interaction_grouped_source_pair_py,
        exchange_coulomb_interaction_py,
    )
except ImportError:
    # Fallback for direct execution/testing
    try:
        from epsilon_q2d import epsilon_q2d_py
        from screened_coulomb_interaction import screened_coulomb_interaction_py
        from Vd_singularity import Vd_singularity_py
        from direct_coulomb_interaction import direct_coulomb_interaction_py
        from folding_Gvector import folding_Gvector_py
        from wave_vector_folding import wave_vector_folding_py
        from bare_coulomb_q2d import bare_coulomb_q2d_py
        from exchange_coulomb_interaction import (
            exchange_coulomb_interaction_batched_j_py,
            exchange_coulomb_interaction_grouped_source_pair_py,
            exchange_coulomb_interaction_py,
        )
    except ImportError:
        print("Error: Cannot import dependencies in 'hbse_worker.py'.", file=sys.stderr)
        sys.exit(1)

def calculate_hbse_column(
    j, N_kpt, Ec_kex, Ev, k, b1, b2, d, q_data, epsQ2D_data,
    Wq0, wi, tau, Basis_tau_index_cell, WF_center, N_atoms,
    Cc_kex, Cv, c, cc, v, vv, e, eps0, exchange_interaction=False, chi_p=None, R_mNN=None, verbose=False, G_dir_max=None,
    return_timing=False
):
    """
    calculate_hbse_column
    
    PURPOSE:
        Calculates a single column 'j' of a block of the HBSE matrix.
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        j (int): The 0-based k-point index for this column.
        (all other args): All necessary data pre-loaded by BSE.py.
        verbose (bool): If True, print debug info.
    
    OUTPUT:
        tuple: (KE_j, Vd_col_j)
        KE_j (complex): The single diagonal kinetic energy element.
        Vd_col_j (np.ndarray): The full (unweighted) Vd column vector (N_kpt,).
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    if verbose:
        print(f"  [Worker] Starting column j={j} for block ({vv,cc}, {v,c})...")

    try:
        # 1. Calculate Kinetic Energy (Diagonal Term)
        # Note: We use 0-based indexing for arrays (j),
        # but 1-based indexing for bands (c, v, etc.)
        KE_j = (Ec_kex[j, c-1] - Ev[j, v-1]) * (c == cc) * (v == vv)

        # 2. Calculate q = k_i - k_j for all i
        if return_timing:
            vd_start = time.perf_counter()
        q = k - k[j, :] # Shape (N_kpt, 2)

        # 3. Calculate Screened Coulomb Interaction
        Vq, epsilon_q = screened_coulomb_interaction_py(
            q, b1, b2, d, q_data, epsQ2D_data, G_dir_max=G_dir_max, verbose=verbose
        )

        # 4. Calculate full Vd column
        Vd_col_j = direct_coulomb_interaction_py(
            Vq, epsilon_q, k, b1, b2, Wq0[j], wi[j], q,
            tau, Basis_tau_index_cell, WF_center, N_atoms,
            Cc_kex, Cv, c, cc, v, vv, j, e, eps0, verbose=verbose, G_dir_max=G_dir_max
        )
        vd_time = time.perf_counter() - vd_start if return_timing else 0.0

        Vx_col_j = np.zeros(N_kpt, dtype=complex)
        vx_time = 0.0
        if exchange_interaction:
            if chi_p is None or R_mNN is None:
                raise ValueError("Exchange interaction requested but chi_p/R_mNN were not provided.")
            if return_timing:
                vx_start = time.perf_counter()
            Vx_col_j = exchange_coulomb_interaction_py(
                chi_p, R_mNN, k, N_kpt, Cv, Cc_kex, c, cc, v, vv, j
            )
            vx_time = time.perf_counter() - vx_start if return_timing else 0.0

        if verbose:
            print(f"  [Worker] Finished column j={j}.")

        if return_timing:
            return KE_j, Vd_col_j + Vx_col_j, {"Time_Vd": vd_time, "Time_Vx": vx_time}
        return KE_j, Vd_col_j + Vx_col_j

    except Exception as e:
        print(f"--- ERROR in Python Worker (j={j}) ---", file=sys.stderr)
        print(f"Exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Return zeros to avoid crashing the whole pool
        if return_timing:
            return 0.0, np.zeros(N_kpt, dtype=complex), {"Time_Vd": 0.0, "Time_Vx": 0.0}
        return 0.0, np.zeros(N_kpt, dtype=complex)


def calculate_hbse_chunk(
    j_indices, N_kpt, Ec_kex, Ev, k, b1, b2, d, q_data, epsQ2D_data,
    Wq0, wi, tau, Basis_tau_index_cell, WF_center, N_atoms,
    Cc_kex, Cv, c, cc, v, vv, e, eps0, exchange_interaction=False, chi_p=None, R_mNN=None, verbose=False, G_dir_max=None,
    return_timing=False
):
    """
    calculate_hbse_chunk
    
    PURPOSE:
        Calculate ordered HBSE interaction columns for one bounded j chunk.
    
    DESCRIPTION:
        Direct and screened terms intentionally retain the scalar reference calls.
        Exchange is evaluated once for the complete chunk. The returned J indices
        are the unmodified input order, allowing deterministic parent-side scatter.
    
    INPUT ARGUMENTS:
        See the annotated signature and detailed description.
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    js = np.asarray(j_indices, dtype=np.intp)
    if js.ndim != 1 or js.size == 0:
        raise ValueError("j_indices must be a nonempty one-dimensional sequence")
    if np.any(js < 0) or np.any(js >= N_kpt):
        raise ValueError("j_indices contain an out-of-range k-point")

    if verbose:
        print(f"  [Worker] Starting chunk j={js[0]}..{js[-1]} for block ({vv, cc}, {v, c})...")

    try:
        interaction = np.empty((N_kpt, js.size), dtype=np.complex128)
        vd_time = 0.0
        vx_time = 0.0
        for column, j in enumerate(js):
            if return_timing:
                vd_start = time.perf_counter()
            q = k - k[j, :]
            Vq, epsilon_q = screened_coulomb_interaction_py(
                q, b1, b2, d, q_data, epsQ2D_data, G_dir_max=G_dir_max, verbose=verbose
            )
            interaction[:, column] = direct_coulomb_interaction_py(
                Vq, epsilon_q, k, b1, b2, Wq0[j], wi[j], q,
                tau, Basis_tau_index_cell, WF_center, N_atoms,
                Cc_kex, Cv, c, cc, v, vv, int(j), e, eps0, verbose=verbose, G_dir_max=G_dir_max
            )
            if return_timing:
                vd_time += time.perf_counter() - vd_start

        if exchange_interaction:
            if chi_p is None or R_mNN is None:
                raise ValueError("Exchange interaction requested but chi_p/R_mNN were not provided.")
            if return_timing:
                vx_start = time.perf_counter()
            interaction += exchange_coulomb_interaction_batched_j_py(
                chi_p, R_mNN, k, N_kpt, Cv, Cc_kex, c, cc, v, vv, js
            )
            if return_timing:
                vx_time = time.perf_counter() - vx_start

        if verbose:
            print(f"  [Worker] Finished chunk j={js[0]}..{js[-1]}.")
        if return_timing:
            return js, interaction, {"Time_Vd": vd_time, "Time_Vx": vx_time}
        return js, interaction

    except Exception as error:
        print(f"--- ERROR in Python Chunk Worker (j={js[0]}..{js[-1]}) ---", file=sys.stderr)
        print(f"Exception: {error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if return_timing:
            return js, np.zeros((N_kpt, js.size), dtype=np.complex128), {"Time_Vd": 0.0, "Time_Vx": 0.0}
        return js, np.zeros((N_kpt, js.size), dtype=np.complex128)


def _screened_from_qmag(q_mag, d, q_data, epsQ2D_data):
    """
    _screened_from_qmag
    
    PURPOSE:
        Screening helper using an already-folded q magnitude.
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        See the annotated signature and detailed description.
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    epsilon_q = epsilon_q2d_py(q_data, epsQ2D_data, q_mag)
    return bare_coulomb_q2d_py(q_mag, d), epsilon_q


def _expigtau_from_min(N_kpt, NB, Gi, MinNdkG, tau, Basis_tau_index_cell, N_atoms):
    """
    _expigtau_from_min
    
    PURPOSE:
        Expigtau from min.
    
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
    Gkkp = Gi[MinNdkG - 1, :]
    Gkkp_3d = np.hstack([Gkkp, np.zeros((N_kpt, 1))])
    expiGtau = np.zeros((N_kpt, NB), dtype=np.complex128)
    for Ni in range(N_atoms):
        basis_1based = Basis_tau_index_cell[Ni]
        if not basis_1based:
            continue
        basis_0based = np.asarray(basis_1based, dtype=np.intp) - 1
        phase_vec = np.exp(1j * np.sum(Gkkp_3d * tau[Ni, :][np.newaxis, :], axis=1))
        expiGtau[:, basis_0based] = phase_vec[:, np.newaxis]
    return expiGtau


def _direct_vd_from_prepared(
    Vq, epsilon_q, Wq0_j, wi_j, Cc_kex_tilde, Cv_tilde, expiGtau,
    c, cc, v, vv, j, e, eps0
):
    """
    _direct_vd_from_prepared
    
    PURPOSE:
        Direct vd from prepared.
    
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
    Cj_c_vec = Cc_kex_tilde[j, :, cc - 1][np.newaxis, :]
    Cj_v_vec = Cv_tilde[j, :, vv - 1][np.newaxis, :]
    Mc = np.sum(
        np.conj(Cc_kex_tilde[:, :, c - 1]) * expiGtau * Cj_c_vec,
        axis=1
    )
    Mv = np.sum(
        np.conj(Cv_tilde[:, :, v - 1]) * expiGtau * Cj_v_vec,
        axis=1
    )
    Vd = -((e**2) / (4 * np.pi * eps0)) * (Vq / epsilon_q) * Mc * np.conj(Mv)
    Vd[j] = Vd_singularity_py(
        e, eps0, Wq0_j, wi_j, Cc_kex_tilde, Cv_tilde, c, cc, v, vv, j
    )
    return Vd


def calculate_vd_grouped_chunk(
    j_indices, block_specs, N_kpt, k, b1, b2, d, q_data, epsQ2D_data,
    Wq0, wi, tau, Basis_tau_index_cell, N_atoms,
    Cc_kex_tilde, Cv_tilde, e, eps0, verbose=False, return_timing=False, G_dir_max=None
):
    """
    calculate_vd_grouped_chunk
    
    PURPOSE:
        Calculate Vd for all upper-triangular band blocks for one source chunk.
    
    DESCRIPTION:
        This worker implements only the validated Vd reuse scope: precomputed
        phase-shifted coefficients and per-source folded q/screening/G phases.
    
    INPUT ARGUMENTS:
        See the annotated signature and detailed description.
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    js = np.asarray(j_indices, dtype=np.intp)
    if js.ndim != 1 or js.size == 0:
        raise ValueError("j_indices must be a nonempty one-dimensional sequence")
    if np.any(js < 0) or np.any(js >= N_kpt):
        raise ValueError("j_indices contain an out-of-range k-point")

    block_specs = tuple(block_specs)
    NB = Cc_kex_tilde.shape[1]
    try:
        if verbose:
            print(f"  [Worker] Starting grouped Vd chunk j={js[0]}..{js[-1]} with {len(block_specs)} blocks...")
        task_start = time.perf_counter()
        Gi = folding_Gvector_py(b1, b2, G_dir_max)
        grouped = np.empty((len(block_specs), N_kpt, js.size), dtype=np.complex128)
        prep_time = 0.0
        reduction_time = 0.0

        for column, j_raw in enumerate(js):
            j = int(j_raw)
            prep_start = time.perf_counter()
            q = k - k[j, :]
            q_mag, MinNdkG = wave_vector_folding_py(q, b1, b2, G_dir_max)
            Vq, epsilon_q = _screened_from_qmag(q_mag, d, q_data, epsQ2D_data)
            expiGtau = _expigtau_from_min(
                N_kpt, NB, Gi, MinNdkG, tau, Basis_tau_index_cell, N_atoms
            )
            prep_time += time.perf_counter() - prep_start

            reduction_start = time.perf_counter()
            for block_pos, block in enumerate(block_specs):
                grouped[block_pos, :, column] = _direct_vd_from_prepared(
                    Vq, epsilon_q, Wq0[j], wi[j],
                    Cc_kex_tilde, Cv_tilde, expiGtau,
                    int(block["c"]), int(block["cc"]), int(block["v"]), int(block["vv"]),
                    j, e, eps0
                )
            reduction_time += time.perf_counter() - reduction_start

        vd_time = time.perf_counter() - task_start
        if verbose:
            print(f"  [Worker] Finished grouped Vd chunk j={js[0]}..{js[-1]}.")
        if return_timing:
            return js, block_specs, grouped, {
                "Time_Vd": vd_time,
                "Time_Vd_grouped_task": vd_time,
                "Time_Vd_grouped_prep": prep_time,
                "Time_Vd_grouped_reduction": reduction_time,
            }
        return js, block_specs, grouped

    except Exception as error:
        print(f"--- ERROR in Python Grouped Vd Worker (j={js[0]}..{js[-1]}) ---", file=sys.stderr)
        print(f"Exception: {error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if return_timing:
            return js, block_specs, np.zeros((len(block_specs), N_kpt, js.size), dtype=np.complex128), {
                "Time_Vd": 0.0,
                "Time_Vd_grouped_task": 0.0,
                "Time_Vd_grouped_prep": 0.0,
                "Time_Vd_grouped_reduction": 0.0,
            }
        return js, block_specs, np.zeros((len(block_specs), N_kpt, js.size), dtype=np.complex128)


def calculate_vx_chunk(
    j_indices, N_kpt, k, Cv, Cc_kex, c, cc, v, vv,
    exchange_interaction=False, chi_p=None, R_mNN=None, return_timing=False
):
    """
    calculate_vx_chunk
    
    PURPOSE:
        Evaluate the existing block-local Vx path without changing its inputs.
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        See the annotated signature and detailed description.
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    js = np.asarray(j_indices, dtype=np.intp)
    if js.ndim != 1 or js.size == 0:
        raise ValueError("j_indices must be a nonempty one-dimensional sequence")
    start = time.perf_counter()
    if not exchange_interaction:
        vx = np.zeros((N_kpt, js.size), dtype=np.complex128)
    else:
        if chi_p is None or R_mNN is None:
            raise ValueError("Exchange interaction requested but chi_p/R_mNN were not provided.")
        vx = exchange_coulomb_interaction_batched_j_py(
            chi_p, R_mNN, k, N_kpt, Cv, Cc_kex, c, cc, v, vv, js
        )
    vx_time = time.perf_counter() - start
    if return_timing:
        return js, vx, {"Time_Vx": vx_time}
    return js, vx


# ==========================================================================
# CALCULATE_VX_SOURCE_GROUP_CHUNK
#
# DESCRIPTION:
#   Joblib worker entry point for all target blocks sharing one source band
#   pair and an ordered source-j chunk.
#
# INPUT ARGUMENTS:
#   chi_packed_descriptor : optional read-only packed-chi descriptor created
#       by the parent process; chi_p is omitted from worker transport when it
#       is supplied.
#   target_blocks, j_indices : source/target mapping metadata and ordered
#       chunk indices used unchanged by parent-side HBSE scattering.
#
# OUTPUT:
#   results : Vx blocks and optional timing/profile metadata for each target.
#
# METHODOLOGY:
#   Delegates the validated packed-S1 and fused-grouped-reduction contractions
#   to the exchange kernel.  A persistent loky pool lets each worker reuse its
#   read-only packed map across the four source-group batches.
#
# NOTES:
#   No worker repacks chi.  Errors retain the existing zero-block fallback.
# ==========================================================================
def calculate_vx_source_group_chunk(
    j_indices, target_blocks, N_kpt, k, Cv, Cc_kex, vv, cc,
    chi_p=None, R_mNN=None, return_timing=False, chi_packed_descriptor=None
):
    """
    calculate_vx_source_group_chunk
    
    PURPOSE:
        Calculate vx source group chunk.
    
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
    js = np.asarray(j_indices, dtype=np.intp)
    if js.ndim != 1 or js.size == 0:
        raise ValueError("j_indices must be a nonempty one-dimensional sequence")
    if np.any(js < 0) or np.any(js >= N_kpt):
        raise ValueError("j_indices contain an out-of-range k-point")
    if not target_blocks:
        raise ValueError("target_blocks must be nonempty")
    if R_mNN is None or (chi_p is None and chi_packed_descriptor is None):
        raise ValueError("Exchange interaction requested but chi_p or packed chi descriptor and R_mNN were not provided.")

    try:
        start = time.perf_counter()
        vx_profile = {} if return_timing else None
        vx_by_target = exchange_coulomb_interaction_grouped_source_pair_py(
            chi_p, R_mNN, k, N_kpt, Cv, Cc_kex, vv, cc, target_blocks, js,
            profile=vx_profile, chi_packed_descriptor=chi_packed_descriptor
        )
        vx_time = time.perf_counter() - start

        results = []
        for target_index, block in enumerate(target_blocks):
            if return_timing:
                results.append((
                    block, js, vx_by_target[target_index],
                    {"Time_Vx": vx_time if target_index == 0 else 0.0,
                     "Vx_profile": vx_profile if target_index == 0 else {}}
                ))
            else:
                results.append((block, js, vx_by_target[target_index]))
        return results

    except Exception as error:
        print(f"--- ERROR in Python Grouped Vx Worker (j={js[0]}..{js[-1]}) ---", file=sys.stderr)
        print(f"Exception: {error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        results = []
        for block in target_blocks:
            zeros = np.zeros((N_kpt, js.size), dtype=np.complex128)
            if return_timing:
                results.append((block, js, zeros, {"Time_Vx": 0.0}))
            else:
                results.append((block, js, zeros))
        return results
