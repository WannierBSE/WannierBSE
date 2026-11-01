"""
build_HBSE

PURPOSE:
    Builds the full HBSE matrix in Python using joblib for parallel

DESCRIPTION:
    processing.

    This module is the Python replacement for build_HBSE.m.
    It imports the worker function from hbse_worker.py and parallelizes
    its execution over all k-points.

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
from joblib import Parallel, delayed
import os # Import os for path joining

HBSE_J_CHUNK_SIZE = 8

# Import the worker function
try:
    # Use relative import for package structure
    from .hbse_worker import (
        calculate_hbse_chunk,
        calculate_hbse_column,
        calculate_vd_grouped_chunk,
        calculate_vx_chunk,
        calculate_vx_source_group_chunk,
    )
    from .Direct.phase_tau_in_Vd import phase_tau_in_Vd_py
except ImportError:
    # Fallback for direct execution/testing
    try:
        from hbse_worker import (
            calculate_hbse_chunk,
            calculate_hbse_column,
            calculate_vd_grouped_chunk,
            calculate_vx_chunk,
            calculate_vx_source_group_chunk,
        )
        from phase_tau_in_Vd import phase_tau_in_Vd_py
    except ImportError:
        print("Error: Cannot import 'calculate_hbse_column' from 'hbse_worker.py'.", file=sys.stderr)
        sys.exit(1)


def _hbse_block_slices(c, v, cc, vv, Nc, N_kpt):
    """
    _hbse_block_slices
    
    PURPOSE:
        Hbse block slices.
    
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
    idx1_start = ((c - 1) + (v - 1) * Nc) * N_kpt
    idx1_end = (c + (v - 1) * Nc) * N_kpt
    idx2_start = ((cc - 1) + (vv - 1) * Nc) * N_kpt
    idx2_end = (cc + (vv - 1) * Nc) * N_kpt
    return slice(idx1_start, idx1_end), slice(idx2_start, idx2_end), idx2_start


def _scatter_block_columns(HBSE, block_values, j_chunk, c, v, cc, vv, Nc, N_kpt):
    """
    _scatter_block_columns
    
    PURPOSE:
        Scatter block columns.
    
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
    idx1_slice, _idx2_slice, idx2_start = _hbse_block_slices(c, v, cc, vv, Nc, N_kpt)
    col_indices = idx2_start + j_chunk
    HBSE[idx1_slice, col_indices] += block_values
    if v != vv or c != cc:
        row_indices = idx2_start + j_chunk
        HBSE[row_indices, idx1_slice] += block_values.conj().T


def _build_exchange_source_groups(block_specs, total_blocks, j_chunks):
    """
    _build_exchange_source_groups
    
    PURPOSE:
        Build exchange source groups.
    
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
    groups = []
    for source_index in range(total_blocks):
        source_blocks = [
            block for block in block_specs
            if int(block["source_block"]) == source_index
        ]
        if not source_blocks:
            continue
        vv = int(source_blocks[0]["vv"])
        cc = int(source_blocks[0]["cc"])
        for j_chunk in j_chunks:
            groups.append((source_index, vv, cc, source_blocks, j_chunk))
    return groups

# ==========================================================================
# BUILD_HBSE_PY
#
# DESCRIPTION:
#   Constructs the unweighted complex HBSE interaction matrix using grouped
#   direct and exchange worker tasks.
#
# INPUT ARGUMENTS:
#   chi_p, R_mNN : precomputed exchange tensor and real-space vectors when
#       exchange is enabled.
#   chi_packed_descriptor : optional parent-created, read-only packed-chi
#       descriptor passed to Vx workers instead of serializing packed data.
#
# OUTPUT:
#   HBSE or (HBSE, timing) when return_timing is requested.
#
# METHODOLOGY:
#   Retains four source-group Vx batches, chunk size, scattering, and all
#   physical contractions.  A persistent loky pool spans the batches so the
#   validated packed chi map is opened once per worker and reused.
#
# NOTES:
#   This routine does not create or remove the backing file; the calling
#   driver owns its lifecycle.  No task-topology redesign is introduced.
# ==========================================================================
def build_hbse_py(
    vc_band_BSE, Nv, Nc, N_kpt, Number_par,
    Ec_kex, Ev, k_np, b1, b2, d, q_data, epsQ2D_data,
    Wq0, wi, tau, Basis_tau_index_cell, WF_center, N_atoms,
    Cc_kex, Cv, e, eps0, exchange_interaction=False, chi_p=None, R_mNN=None, verbose=False,
    return_timing=False, G_dir_max=None, chi_packed_descriptor=None
):
    """
    build_hbse_py
    
    PURPOSE:
        Builds the full HBSE matrix in Python using joblib.
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        (all args): All necessary data pre-loaded by BSE.py.
        Number_par (int): Number of parallel workers (passed to n_jobs).
        verbose (bool): If True, print debug info.
    
    OUTPUT:
        np.ndarray: The full, complex HBSE matrix.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """

    # --- DEBUG SETTINGS ---
    # Set to a k-point index (e.g., 0) to debug, or None for full run.
    debug_k_point = None
    # Set to a block (e.g., [0,0]) to debug, or None for full run.
    # debug_block = [0, 0] # vvcci=0, vci=0
    debug_block = None
    # --- END DEBUG SETTINGS ---

    # --- MODIFICATION: Define output dir for debug file ---
    # Assumes execution from root, same as BSE.py
    debug_output_dir = 'Exciton_data'
    # --- END MODIFICATION ---

    HBSE_size = Nc * Nv * N_kpt
    # Note: We create a complex matrix, as Vd is complex
    HBSE = np.zeros((HBSE_size, HBSE_size), dtype=complex)
    timing = {
        "Time_Vd_worker_accumulated": 0.0,
        "Time_Vd_grouped_task_accumulated": 0.0,
        "Time_Vd_grouped_prep_accumulated": 0.0,
        "Time_Vd_grouped_reduction_accumulated": 0.0,
        "Time_Vx_worker_accumulated": 0.0,
        "Time_Vx_parallel_dispatch_s": 0.0,
        "Vx_profile": {},
    }

    def merge_vx_profile(source):
        """
        merge_vx_profile
        
        PURPOSE:
            Merge vx profile.
        
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
        destination = timing["Vx_profile"]
        for key, value in (source or {}).items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                destination[key] = destination.get(key, 0) + value

    total_blocks = len(vc_band_BSE)
    processed_blocks = 0

    for valence, conduction in vc_band_BSE:
        print(f"  -> Processing subspace: v{int(valence)}, c{int(conduction)}", flush=True)

    if verbose:
        print(f"  [Debug] build_hbse_py: Starting matrix construction...")
        print(f"  [Debug] build_hbse_py: Total k-points N_kpt = {N_kpt}")
        print(f"  [Debug] build_hbse_py: Total size = {HBSE_size} x {HBSE_size}")
        print(f"  [Debug] build_hbse_py: Using {Number_par} parallel workers (n_jobs).")

    try:
        if debug_k_point is None and debug_block is None:
            block_specs = []
            for vvcci in range(total_blocks):
                vv = int(vc_band_BSE[vvcci, 0])
                cc = int(vc_band_BSE[vvcci, 1])
                for vci in range(vvcci, total_blocks):
                    v = int(vc_band_BSE[vci, 0])
                    c = int(vc_band_BSE[vci, 1])
                    block_specs.append({
                        "block_index": len(block_specs),
                        "source_block": vvcci,
                        "target_block": vci,
                        "vv": vv,
                        "cc": cc,
                        "v": v,
                        "c": c,
                    })

            j_chunks = [
                np.arange(start, min(start + HBSE_J_CHUNK_SIZE, N_kpt), dtype=np.intp)
                for start in range(0, N_kpt, HBSE_J_CHUNK_SIZE)
            ]

            phase_start = time.perf_counter()
            Cc_kex_tilde, Cv_tilde = phase_tau_in_Vd_py(k_np, Cc_kex, Cv, tau, WF_center, verbose=False)
            timing["Time_Vd_phase_precompute"] = time.perf_counter() - phase_start

            if Number_par == 1:
                vd_results = [
                    calculate_vd_grouped_chunk(
                        j_chunk, block_specs, N_kpt, k_np, b1, b2, d, q_data, epsQ2D_data,
                        Wq0, wi, tau, Basis_tau_index_cell, N_atoms,
                        Cc_kex_tilde, Cv_tilde, e, eps0,
                        verbose=False, return_timing=return_timing, G_dir_max=G_dir_max
                    )
                    for j_chunk in j_chunks
                ]
            else:
                vd_results = Parallel(n_jobs=Number_par)(
                    delayed(calculate_vd_grouped_chunk)(
                        j_chunk, block_specs, N_kpt, k_np, b1, b2, d, q_data, epsQ2D_data,
                        Wq0, wi, tau, Basis_tau_index_cell, N_atoms,
                        Cc_kex_tilde, Cv_tilde, e, eps0,
                        verbose=False, return_timing=return_timing, G_dir_max=G_dir_max
                    )
                    for j_chunk in j_chunks
                )

            for result in vd_results:
                if return_timing:
                    j_chunk, returned_blocks, grouped_vd, chunk_timing = result
                    timing["Time_Vd_worker_accumulated"] += float(chunk_timing.get("Time_Vd", 0.0))
                    timing["Time_Vd_grouped_task_accumulated"] += float(chunk_timing.get("Time_Vd_grouped_task", 0.0))
                    timing["Time_Vd_grouped_prep_accumulated"] += float(chunk_timing.get("Time_Vd_grouped_prep", 0.0))
                    timing["Time_Vd_grouped_reduction_accumulated"] += float(chunk_timing.get("Time_Vd_grouped_reduction", 0.0))
                else:
                    j_chunk, returned_blocks, grouped_vd = result
                for block_pos, block in enumerate(returned_blocks):
                    _scatter_block_columns(
                        HBSE, grouped_vd[block_pos], j_chunk,
                        int(block["c"]), int(block["v"]), int(block["cc"]), int(block["vv"]),
                        Nc, N_kpt
                    )

            if exchange_interaction:
                vx_wall_start = time.perf_counter()
                vx_groups = _build_exchange_source_groups(block_specs, total_blocks, j_chunks)
                # Preserve the four source-group barriers.  The pool remains
                # alive across them solely to reuse worker-local chi memmaps.
                def dispatch_vx_groups(vx_parallel=None):
                    for source_index in range(total_blocks):
                        source_groups = [group for group in vx_groups if group[0] == source_index]
                        vx_dispatch_start = time.perf_counter()
                        if vx_parallel is None:
                            vx_results = [
                                calculate_vx_source_group_chunk(
                                    j_chunk, target_blocks, N_kpt, k_np, Cv, Cc_kex, vv, cc,
                                    chi_p=None if chi_packed_descriptor is not None else chi_p,
                                    R_mNN=R_mNN, return_timing=return_timing,
                                    chi_packed_descriptor=chi_packed_descriptor
                                )
                                for _source_index, vv, cc, target_blocks, j_chunk in source_groups
                            ]
                        else:
                            vx_results = vx_parallel(
                                delayed(calculate_vx_source_group_chunk)(
                                    j_chunk, target_blocks, N_kpt, k_np, Cv, Cc_kex, vv, cc,
                                    chi_p=None if chi_packed_descriptor is not None else chi_p,
                                    R_mNN=R_mNN, return_timing=return_timing,
                                    chi_packed_descriptor=chi_packed_descriptor
                                )
                                for _source_index, vv, cc, target_blocks, j_chunk in source_groups
                            )
                        timing["Time_Vx_parallel_dispatch_s"] += time.perf_counter() - vx_dispatch_start
                        for group_result in vx_results:
                            for result in group_result:
                                if return_timing:
                                    block, j_chunk, vx_block, chunk_timing = result
                                    timing["Time_Vx_worker_accumulated"] += float(chunk_timing.get("Time_Vx", 0.0))
                                    merge_vx_profile(chunk_timing.get("Vx_profile"))
                                else:
                                    block, j_chunk, vx_block = result
                                _scatter_block_columns(
                                    HBSE, vx_block, j_chunk,
                                    int(block["c"]), int(block["v"]), int(block["cc"]), int(block["vv"]),
                                    Nc, N_kpt
                                )
                if Number_par == 1:
                    dispatch_vx_groups()
                else:
                    with Parallel(n_jobs=Number_par, backend="loky") as vx_parallel:
                        dispatch_vx_groups(vx_parallel)
                timing["Time_Vx_wall_elapsed"] = time.perf_counter() - vx_wall_start

            if verbose:
                print(f"  [Debug] build_hbse_py: Grouped Vd matrix construction complete.")

            if return_timing:
                return HBSE, timing
            return HBSE

        # Loop over all (v,c) pairs for the *rows* of the block matrix
        for vvcci in range(total_blocks):
            vv = vc_band_BSE[vvcci, 0] # 1-based row-band-v
            cc = vc_band_BSE[vvcci, 1] # 1-based row-band-c

            # Loop over all (v,c) pairs for the *columns* (from vvcci onwards)
            for vci in range(vvcci, total_blocks):
                v = vc_band_BSE[vci, 0] # 1-based col-band-v
                c = vc_band_BSE[vci, 1] # 1-based col-band-c

                # --- DEBUG BLOCK ---
                if debug_block is not None and (vvcci != debug_block[0] or vci != debug_block[1]):
                    continue
                # --- END DEBUG BLOCK ---

                block_start_time = time.perf_counter()
                if verbose or debug_block is not None:
                    print(f"--- Processing Block (vv,cc)={vv,cc} <-> (v,c)={v,c} ---")

                # --- Parallel Execution (Joblib) ---

                # --- DEBUG BLOCK ---
                if debug_k_point is not None:
                    print(f"--- !!! DEBUG MODE: Running only for k-point j={debug_k_point} !!! ---")
                    # Run for only one k-point, serially
                    results = [
                        calculate_hbse_column(
                            debug_k_point, N_kpt, Ec_kex, Ev, k_np, b1, b2, d, q_data, epsQ2D_data,
                            Wq0, wi, tau, Basis_tau_index_cell, WF_center, N_atoms,
                            Cc_kex, Cv, c, cc, v, vv, e, eps0,
                            exchange_interaction=exchange_interaction, chi_p=chi_p, R_mNN=R_mNN,
                            verbose=True, return_timing=return_timing, G_dir_max=G_dir_max
                        )
                    ]
                    # Manually save the intermediate Vd result
                    if return_timing:
                        KE_j_debug, interaction_col_debug, timing_debug = results[0]
                    else:
                        KE_j_debug, interaction_col_debug = results[0]
                    interaction_to_save = interaction_col_debug.reshape(-1, 1) # Make it a column
                    Vd_complex = np.hstack([interaction_to_save.real, interaction_to_save.imag])
                    header = f"Debug interaction column from Python. (vv,cc)=({vv},{cc}), (v,c)=({v},{c}), j={debug_k_point}. Cols: Real, Imag"

                    # --- MODIFICATION: Save to Exciton_data directory ---
                    os.makedirs(debug_output_dir, exist_ok=True)
                    output_filename = os.path.join(debug_output_dir, f'Vd_py_j{debug_k_point}.txt')
                    np.savetxt(output_filename, Vd_complex, fmt='%.18e', header=header, comments='# ')
                    print(f"--- !!! DEBUG MODE: Saved {output_filename} !!! ---")

                    if return_timing:
                        results = [(np.array([debug_k_point], dtype=np.intp), interaction_col_debug.reshape(-1, 1), timing_debug)]
                    else:
                        results = [(np.array([debug_k_point], dtype=np.intp), interaction_col_debug.reshape(-1, 1))]

                else:
                    j_chunks = [
                        np.arange(start, min(start + HBSE_J_CHUNK_SIZE, N_kpt), dtype=np.intp)
                        for start in range(0, N_kpt, HBSE_J_CHUNK_SIZE)
                    ]
                    if Number_par == 1:
                        results = [
                            calculate_hbse_chunk(
                                j_chunk, N_kpt, Ec_kex, Ev, k_np, b1, b2, d, q_data, epsQ2D_data,
                                Wq0, wi, tau, Basis_tau_index_cell, WF_center, N_atoms,
                                Cc_kex, Cv, c, cc, v, vv, e, eps0,
                                exchange_interaction=exchange_interaction, chi_p=chi_p, R_mNN=R_mNN,
                                verbose=False, return_timing=return_timing, G_dir_max=G_dir_max
                            )
                            for j_chunk in j_chunks
                        ]
                    else:
                        results = Parallel(n_jobs=Number_par)(
                            delayed(calculate_hbse_chunk)(
                                j_chunk, N_kpt, Ec_kex, Ev, k_np, b1, b2, d, q_data, epsQ2D_data,
                                Wq0, wi, tau, Basis_tau_index_cell, WF_center, N_atoms,
                                Cc_kex, Cv, c, cc, v, vv, e, eps0,
                                exchange_interaction=exchange_interaction, chi_p=chi_p, R_mNN=R_mNN,
                                verbose=False, return_timing=return_timing, G_dir_max=G_dir_max
                            )
                            for j_chunk in j_chunks
                        )
                # --- END DEBUG BLOCK ---

                # Scatter each chunk by its explicit source-j indices. This is
                # independent of task completion order and preserves final B=6.
                HBSE_Vd_block = np.zeros((N_kpt, N_kpt), dtype=np.complex128)
                for result in results:
                    if return_timing:
                        j_chunk, interaction_block, chunk_timing = result
                        timing["Time_Vd_worker_accumulated"] += float(chunk_timing.get("Time_Vd", 0.0))
                        timing["Time_Vx_worker_accumulated"] += float(chunk_timing.get("Time_Vx", 0.0))
                    else:
                        j_chunk, interaction_block = result
                    HBSE_Vd_block[:, j_chunk] = interaction_block

                # --- Assign the block to the full HBSE matrix ---
                # idx1 (rows) = inner loop (c, v)
                # idx2 (cols) = outer loop (cc, vv)
                idx1_start = ((c - 1) + (v - 1) * Nc) * N_kpt
                idx1_end = (c + (v - 1) * Nc) * N_kpt
                idx1_slice = slice(idx1_start, idx1_end)

                idx2_start = ((cc - 1) + (vv - 1) * Nc) * N_kpt
                idx2_end = (cc + (vv - 1) * Nc) * N_kpt
                idx2_slice = slice(idx2_start, idx2_end)

                # Fill the block directly (Unweighted Interaction V)
                HBSE[idx1_slice, idx2_slice] = HBSE_Vd_block

                # Symmetrize the off-diagonal block using Hermitian symmetry
                if v != vv or c != cc:
                    # HBSE[idx2, idx1] = HBSE[idx1, idx2]^H
                    HBSE[idx2_slice, idx1_slice] = HBSE_Vd_block.conj().T

                if verbose or debug_block is not None:
                    block_time = time.perf_counter() - block_start_time
                    processed_blocks += 1
                    print(f"--- Block ({vv,cc})-({v,c}) finished in {block_time:.2f}s "
                          f"({processed_blocks} / {total_blocks*(total_blocks+1)//2}) ---")

    except Exception as e:
        print(f"--- ERROR in build_hbse_py ---", file=sys.stderr)
        print(f"Exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"  [Debug] build_hbse_py: Matrix construction complete.")

    if return_timing:
        return HBSE, timing
    return HBSE
