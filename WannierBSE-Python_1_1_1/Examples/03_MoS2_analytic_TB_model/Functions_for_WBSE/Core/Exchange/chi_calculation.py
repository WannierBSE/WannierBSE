"""
chi_calculation

PURPOSE:
    Compute short-range exchange tensor chi from rho.

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

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from .find_j_index_in_rho_integral import find_j_index_in_rho_integral_py
from .xi_integral import (
    build_b_cache_py,
    build_fixed_grid_cubic_operator_py,
    build_fixed_grid_integration_operator_py,
    build_fixed_grid_b_operators_py,
    build_b_cache_operator_py,
    xi_integral_batched_pairs_from_b_cache_operator_py,
    xi_integral_batched_pairs_from_b_cache_direct_py,
    xi_integral_batched_pairs_from_b_cache_py,
)


def _chi_g_contribution(
    gi,
    G,
    rho,
    R_mNN,
    kex,
    tau,
    n_atoms,
    d_max,
    basis_tau_index,
    mmax_tolerance,
    e,
    eps0,
    z_point_t,
    b1,
    coefficient_operator=None,
    integration_operator=None,
    b_operators=None,
):
    """
    _chi_g_contribution
    
    PURPOSE:
        Chi g contribution.
    
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
    nb = rho.shape[0]
    norm = np.linalg.norm(np.asarray(kex)[:2] + G[:2])
    if norm == 0:
        return gi, np.zeros((nb, nb, nb, nb, len(R_mNN), len(R_mNN)), dtype=complex)

    pref = (e**2 / (2 * eps0)) / norm
    if b_operators is None:
        b_operators = build_fixed_grid_b_operators_py(G, kex, z_point_t)
    all_idx = list(range(nb))
    xi_g = np.zeros((nb, nb, nb, nb, len(R_mNN), len(R_mNN)), dtype=complex)
    r2_data = []
    for r2i, R2 in enumerate(R_mNN):
        l_index_tot = find_j_index_in_rho_integral_py(R2, tau, n_atoms, d_max, basis_tau_index, mmax_tolerance)
        rho2 = rho[:, :, :, r2i, gi]
        b_cache = build_b_cache_py(rho2, G, kex, z_point_t, all_idx, l_index_tot) if b_operators is None else build_b_cache_operator_py(rho2, all_idx, l_index_tot, *b_operators)
        r2_data.append((l_index_tot, b_cache))

    for r1i, R1 in enumerate(R_mNN):
        j_index_tot = find_j_index_in_rho_integral_py(R1, tau, n_atoms, d_max, basis_tau_index, mmax_tolerance)
        for r2i, (l_index_tot, b_cache) in enumerate(r2_data):
            if integration_operator is not None:
                xi_g[:, :, :, :, r1i, r2i] = xi_integral_batched_pairs_from_b_cache_direct_py(
                    rho[:, :, :, r1i, gi], z_point_t, nb, b1, all_idx, j_index_tot,
                    all_idx, l_index_tot, b_cache, integration_operator
                )
            elif coefficient_operator is not None:
                xi_g[:, :, :, :, r1i, r2i] = xi_integral_batched_pairs_from_b_cache_operator_py(
                    rho[:, :, :, r1i, gi], z_point_t, nb, b1, all_idx, j_index_tot,
                    all_idx, l_index_tot, b_cache, coefficient_operator
                )
            else:
                xi_g[:, :, :, :, r1i, r2i] = xi_integral_batched_pairs_from_b_cache_py(
                    rho[:, :, :, r1i, gi], z_point_t, nb, b1, all_idx, j_index_tot,
                    all_idx, l_index_tot, b_cache
                )
    return gi, pref * xi_g


def chi_calculation_py(
    rho,
    R_mNN,
    G_all,
    kex,
    tau,
    n_atoms,
    d_max,
    basis_tau_index,
    mmax_tolerance,
    npar_xi,
    e,
    eps0,
    z_point_t,
    b1,
):
    """
    chi_calculation_py
    
    PURPOSE:
        Chi calculation py.
    
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
    nb = rho.shape[0]
    chi = np.zeros((nb, nb, nb, nb, len(R_mNN), len(R_mNN)), dtype=complex)
    n_workers = max(1, int(npar_xi))
    n_g = len(G_all)
    start = time.perf_counter()
    coefficient_operator = build_fixed_grid_cubic_operator_py(z_point_t)
    integration_operator = build_fixed_grid_integration_operator_py(z_point_t, coefficient_operator)
    print(
        f"[INFO ] Exchange chi: NB={nb}, R={len(R_mNN)}, G={n_g}, "
        f"Nz={rho.shape[2]}, Npar_Xi={n_workers}",
        flush=True,
    )

    if n_workers == 1 or n_g <= 1:
        for gi, G in enumerate(G_all):
            g_start = time.perf_counter()
            print(f"[PROG ] Exchange chi: G {gi + 1}/{n_g} started", flush=True)
            _, chi_g = _chi_g_contribution(
                gi, G, rho, R_mNN, kex, tau, n_atoms, d_max, basis_tau_index,
                mmax_tolerance, e, eps0, z_point_t, b1, coefficient_operator, integration_operator, None
            )
            chi += chi_g
            done = gi + 1
            elapsed = time.perf_counter() - start
            eta = elapsed * (n_g - done) / done if done else 0.0
            print(
                f"[PROG ] Exchange chi: G {done}/{n_g} done "
                f"in {time.perf_counter() - g_start:.2f} s; elapsed {elapsed:.2f} s; ETA {eta:.2f} s",
                flush=True,
            )
        return chi

    print(f"[INFO ] Exchange chi: using ThreadPoolExecutor with {n_workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(
            lambda item: _chi_g_contribution(
                item[0], item[1], rho, R_mNN, kex, tau, n_atoms, d_max, basis_tau_index,
                mmax_tolerance, e, eps0, z_point_t, b1, coefficient_operator, integration_operator, None
            ),
            enumerate(G_all),
        )
        for done, (gi, chi_g) in enumerate(results, start=1):
            chi += chi_g
            elapsed = time.perf_counter() - start
            eta = elapsed * (n_g - done) / done if done else 0.0
            print(
                f"[PROG ] Exchange chi: G {done}/{n_g} collected "
                f"(index {gi + 1}); elapsed {elapsed:.2f} s; ETA {eta:.2f} s",
                flush=True,
            )
    return chi
