"""
rho_calculation

PURPOSE:
    Compute exchange pair-density tensor rho.

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

from __future__ import annotations

import time

import numpy as np

from .find_j_index_in_rho_integral import find_j_index_in_rho_integral_py
from .rho_operator import rho_operator_build_factors_py, rho_operator_evaluate_planes_py
from .shifted_wf import roll_source_indices, shifted_truncated_matrix_direct


def rho_calculation_py(
    W_up,
    W_down,
    W_up_t,
    W_down_t,
    G_all,
    R_mNN,
    n1_n2_mNN,
    zcut_min,
    zcut_max,
    na1,
    na2,
    na3,
    shift_a1,
    shift_a2,
    shift_a3,
    tau,
    n_atoms,
    d_max,
    basis_tau_index,
    mmax_tolerance,
    tin_all,
    nz_t,
    na1_t,
    na2_t,
    kex,
    a1,
    a2,
    u_point_t,
    v_point_t,
    u_t,
    hu,
    hv,
    jac,
    kex_tol,
):
    """
    rho_calculation_py
    
    PURPOSE:
        Rho calculation py.
    
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
    nb = W_up_t.shape[1]
    nz_t = int(nz_t)
    na1_t, na2_t = int(na1_t), int(na2_t)
    plane_t = na1_t * na2_t
    rho = np.zeros((nb, nb, int(zcut_max - zcut_min + 1), len(R_mNN), len(G_all)), dtype=np.complex128)
    tin_all = np.asarray(tin_all, dtype=np.int64)
    dims = (int(na1), int(na2), int(na3))
    start = time.perf_counter()
    factors = rho_operator_build_factors_py(
        G_all, kex, a1, a2, np.asarray(u_t)[:plane_t], u_point_t,
        v_point_t, kex_tol, hu, hv, jac,
    )
    source_indices = []
    for n1, n2 in n1_n2_mNN:
        shift = (int(n1 * shift_a1), int(n2 * shift_a2), int(shift_a3))
        source_indices.append(roll_source_indices(tin_all, dims, shift))
    print(f"[INFO ] Exchange rho: separable factors built in {factors.build_seconds:.3f} s", flush=True)
    for ri, R in enumerate(R_mNN):
        r_start = time.perf_counter()
        print(f"[PROG ] Exchange rho: R {ri + 1}/{len(R_mNN)} started", flush=True)
        Wl_up_t = shifted_truncated_matrix_direct(W_up, source_indices[ri])
        Wl_down_t = shifted_truncated_matrix_direct(W_down, source_indices[ri])
        j_index_tot = find_j_index_in_rho_integral_py(
            R, tau, n_atoms, d_max, basis_tau_index, mmax_tolerance
        )
        pair_tasks = [(jj, j) for jj in range(nb) for j in j_index_tot[jj]]
        report_every = max(1, len(pair_tasks) // 10)
        for pair_count, (jj, j) in enumerate(pair_tasks, start=1):
            wij = (
                np.conj(W_up_t[:, jj]) * Wl_up_t[:, j]
                + np.conj(W_down_t[:, jj]) * Wl_down_t[:, j]
            )
            W_znm = wij.reshape(nz_t, plane_t).reshape(
                nz_t, na2_t, na1_t
            ).transpose(0, 2, 1)
            rho[jj, j, :, ri, :] = rho_operator_evaluate_planes_py(
                factors.U, factors.V, W_znm
            )
            if pair_count == len(pair_tasks) or pair_count % report_every == 0:
                print(
                    f"[PROG ] Exchange rho: R {ri + 1}/{len(R_mNN)} basis pairs "
                    f"{pair_count}/{len(pair_tasks)} done", flush=True,
                )
        elapsed = time.perf_counter() - start
        eta = elapsed * (len(R_mNN) - ri - 1) / (ri + 1)
        print(
            f"[PROG ] Exchange rho: R {ri + 1}/{len(R_mNN)} done "
            f"in {time.perf_counter() - r_start:.2f} s; elapsed {elapsed:.2f} s; ETA {eta:.2f} s",
            flush=True,
        )
    return rho
