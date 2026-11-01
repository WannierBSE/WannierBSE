"""
xi_integral

PURPOSE:
    Z-direction Xi contraction for exchange chi.

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

import numpy as np
from scipy.interpolate import CubicSpline
from .spline_integral_inplane import poly_int0
from .spline_integral_z import poly_int_exp_real


def xi_integral_py(
    rho1,
    rho2,
    G,
    kex,
    z_point_t,
    nb,
    b1,
    jj_index,
    j_index_tot,
    ll_index,
    l_index_tot,
):
    """
    xi_integral_py
    
    PURPOSE:
        Xi integral py.
    
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
    hz = float(z_point_t[1] - z_point_t[0])
    ng = float(np.linalg.norm(np.asarray(G)[:2] + np.asarray(kex)[:2]))
    xi = np.zeros((nb, nb, nb, nb), dtype=complex)
    for ll in ll_index:
        for l in l_index_tot[ll]:
            vals = rho2[ll, l, :]
            c = CubicSpline(z_point_t, vals, bc_type="not-a-knot").c.T
            b_small = np.zeros(rho2.shape[2], dtype=complex)
            b_large = np.zeros(rho2.shape[2], dtype=complex)
            for zi in range(rho2.shape[2] - 1):
                b_small[zi + 1] = np.sum(poly_int_exp_real(c[:zi + 1, 0], c[:zi + 1, 1], c[:zi + 1, 2], c[:zi + 1, 3], -ng, z_point_t[:zi + 1], hz, z_point_t[zi + 1]))
                b_large[zi] = np.sum(poly_int_exp_real(c[zi:, 0], c[zi:, 1], c[zi:, 2], c[zi:, 3], ng, z_point_t[zi:-1], hz, z_point_t[zi]))
            for jj in jj_index:
                for j in j_index_tot[jj]:
                    cs = CubicSpline(z_point_t, np.conj(rho1[jj, j, :]) * b_small, bc_type="not-a-knot").c.T
                    cl = CubicSpline(z_point_t, np.conj(rho1[jj, j, :]) * b_large, bc_type="not-a-knot").c.T
                    xi[jj, j, ll, l] = np.sum(poly_int0(cs[:, 0], cs[:, 1], cs[:, 2], cs[:, 3], z_point_t[:-1], hz)) + np.sum(poly_int0(cl[:, 0], cl[:, 1], cl[:, 2], cl[:, 3], z_point_t[:-1], hz))
    return xi


def xi_integral_batched_pairs_py(rho1, rho2, G, kex, z_point_t, nb, b1,
                                 jj_index, j_index_tot, ll_index, l_index_tot):
    """
    xi_integral_batched_pairs_py
    
    PURPOSE:
        Evaluate Xi with the scalar jj/j pair order batched into spline columns.
    
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
    hz = float(z_point_t[1] - z_point_t[0])
    ng = float(np.linalg.norm(np.asarray(G)[:2] + np.asarray(kex)[:2]))
    xi = np.zeros((nb, nb, nb, nb), dtype=complex)

    pair_order = [(jj, j) for jj in jj_index for j in j_index_tot[jj]]
    if not pair_order:
        return xi
    pair_jj = np.asarray([pair[0] for pair in pair_order], dtype=int)
    pair_j = np.asarray([pair[1] for pair in pair_order], dtype=int)
    rho1_pairs = np.conj(rho1[pair_jj, pair_j, :]).T
    z0 = np.asarray(z_point_t)[:-1, None]

    for ll in ll_index:
        for l in l_index_tot[ll]:
            vals = rho2[ll, l, :]
            c = CubicSpline(z_point_t, vals, bc_type="not-a-knot").c.T
            b_small = np.zeros(rho2.shape[2], dtype=complex)
            b_large = np.zeros(rho2.shape[2], dtype=complex)
            for zi in range(rho2.shape[2] - 1):
                b_small[zi + 1] = np.sum(
                    poly_int_exp_real(c[:zi + 1, 0], c[:zi + 1, 1],
                                      c[:zi + 1, 2], c[:zi + 1, 3], -ng,
                                      z_point_t[:zi + 1], hz, z_point_t[zi + 1])
                )
                b_large[zi] = np.sum(
                    poly_int_exp_real(c[zi:, 0], c[zi:, 1], c[zi:, 2],
                                      c[zi:, 3], ng, z_point_t[zi:-1], hz,
                                      z_point_t[zi])
                )

            cs = CubicSpline(
                z_point_t, rho1_pairs * b_small[:, None], axis=0,
                bc_type="not-a-knot",
            ).c
            cl = CubicSpline(
                z_point_t, rho1_pairs * b_large[:, None], axis=0,
                bc_type="not-a-knot",
            ).c
            small = np.sum(poly_int0(cs[0], cs[1], cs[2], cs[3], z0, hz), axis=0)
            large = np.sum(poly_int0(cl[0], cl[1], cl[2], cl[3], z0, hz), axis=0)
            xi[pair_jj, pair_j, ll, l] = small + large
    return xi


def build_b_cache_py(rho2, G, kex, z_point_t, ll_index, l_index_tot):
    """
    build_b_cache_py
    
    PURPOSE:
        Build rho2-derived z-kernel vectors keyed by the original (ll, l) order.
    
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
    hz = float(z_point_t[1] - z_point_t[0])
    ng = float(np.linalg.norm(np.asarray(G)[:2] + np.asarray(kex)[:2]))
    cache = {}

    for ll in ll_index:
        for l in l_index_tot[ll]:
            vals = rho2[ll, l, :]
            c = CubicSpline(z_point_t, vals, bc_type="not-a-knot").c.T
            b_small = np.zeros(rho2.shape[2], dtype=complex)
            b_large = np.zeros(rho2.shape[2], dtype=complex)
            for zi in range(rho2.shape[2] - 1):
                b_small[zi + 1] = np.sum(
                    poly_int_exp_real(c[:zi + 1, 0], c[:zi + 1, 1],
                                      c[:zi + 1, 2], c[:zi + 1, 3], -ng,
                                      z_point_t[:zi + 1], hz, z_point_t[zi + 1])
                )
                b_large[zi] = np.sum(
                    poly_int_exp_real(c[zi:, 0], c[zi:, 1], c[zi:, 2],
                                      c[zi:, 3], ng, z_point_t[zi:-1], hz,
                                      z_point_t[zi])
                )
            cache[(ll, l)] = (b_small, b_large)
    return cache


def xi_integral_batched_pairs_from_b_cache_py(rho1, z_point_t, nb, b1,
                                               jj_index, j_index_tot,
                                               ll_index, l_index_tot, b_cache):
    """
    xi_integral_batched_pairs_from_b_cache_py
    
    PURPOSE:
        Evaluate pair-batched Xi using precomputed rho2 z-kernel vectors.
    
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
    hz = float(z_point_t[1] - z_point_t[0])
    xi = np.zeros((nb, nb, nb, nb), dtype=complex)
    pair_order = [(jj, j) for jj in jj_index for j in j_index_tot[jj]]
    if not pair_order:
        return xi
    pair_jj = np.asarray([pair[0] for pair in pair_order], dtype=int)
    pair_j = np.asarray([pair[1] for pair in pair_order], dtype=int)
    rho1_pairs = np.conj(rho1[pair_jj, pair_j, :]).T
    z0 = np.asarray(z_point_t)[:-1, None]
    for ll in ll_index:
        for l in l_index_tot[ll]:
            b_small, b_large = b_cache[(ll, l)]
            cs = CubicSpline(z_point_t, rho1_pairs * b_small[:, None], axis=0, bc_type="not-a-knot").c
            cl = CubicSpline(z_point_t, rho1_pairs * b_large[:, None], axis=0, bc_type="not-a-knot").c
            small = np.sum(poly_int0(cs[0], cs[1], cs[2], cs[3], z0, hz), axis=0)
            large = np.sum(poly_int0(cl[0], cl[1], cl[2], cl[3], z0, hz), axis=0)
            xi[pair_jj, pair_j, ll, l] = small + large
    return xi



def build_fixed_grid_cubic_operator_py(z_point_t):
    """
    build_fixed_grid_cubic_operator_py
    
    PURPOSE:
        Build the fixed-grid SciPy CubicSpline coefficient response operator.
    
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
    z = np.asarray(z_point_t, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.all(np.diff(z) > 0):
        raise ValueError("z_point_t must be a strictly increasing one-dimensional grid.")
    n_z = z.size
    operator = np.empty((4 * (n_z - 1), n_z), dtype=np.float64)
    for column in range(n_z):
        basis = np.zeros(n_z, dtype=np.float64)
        basis[column] = 1.0
        operator[:, column] = CubicSpline(z, basis, bc_type="not-a-knot").c.reshape(-1)
    operator.setflags(write=False)
    return operator


def xi_integral_batched_pairs_from_b_cache_operator_py(rho1, z_point_t, nb, b1,
                                                        jj_index, j_index_tot,
                                                        ll_index, l_index_tot,
                                                        b_cache, coefficient_operator):
    """
    xi_integral_batched_pairs_from_b_cache_operator_py
    
    PURPOSE:
        Evaluate pair-batched Xi using cached b vectors and fixed-grid coefficients.
    
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
    n_z = len(z_point_t)
    operator = np.asarray(coefficient_operator, dtype=np.float64)
    expected_shape = (4 * (n_z - 1), n_z)
    if operator.shape != expected_shape:
        raise ValueError(f"Invalid fixed-grid coefficient operator shape {operator.shape}; expected {expected_shape}.")

    hz = float(z_point_t[1] - z_point_t[0])
    xi = np.zeros((nb, nb, nb, nb), dtype=complex)
    pair_order = [(jj, j) for jj in jj_index for j in j_index_tot[jj]]
    if not pair_order:
        return xi
    pair_jj = np.asarray([pair[0] for pair in pair_order], dtype=int)
    pair_j = np.asarray([pair[1] for pair in pair_order], dtype=int)
    rho1_pairs = np.conj(rho1[pair_jj, pair_j, :]).T
    z0 = np.asarray(z_point_t)[:-1, None]

    for ll in ll_index:
        for l in l_index_tot[ll]:
            b_small, b_large = b_cache[(ll, l)]
            cs = (operator @ (rho1_pairs * b_small[:, None])).reshape(4, n_z - 1, len(pair_order))
            cl = (operator @ (rho1_pairs * b_large[:, None])).reshape(4, n_z - 1, len(pair_order))
            small = np.sum(poly_int0(cs[0], cs[1], cs[2], cs[3], z0, hz), axis=0)
            large = np.sum(poly_int0(cl[0], cl[1], cl[2], cl[3], z0, hz), axis=0)
            xi[pair_jj, pair_j, ll, l] = small + large
    return xi



def build_fixed_grid_integration_operator_py(z_point_t, coefficient_operator=None):
    """
    build_fixed_grid_integration_operator_py
    
    PURPOSE:
        Build the fixed-grid Y-to-integral response using the H reference path.
    
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
    z = np.asarray(z_point_t, dtype=float)
    h = build_fixed_grid_cubic_operator_py(z) if coefficient_operator is None else np.asarray(coefficient_operator, dtype=np.float64)
    n_z = z.size
    if h.shape != (4 * (n_z - 1), n_z):
        raise ValueError("Invalid fixed-grid coefficient operator for z_point_t.")
    hz = float(z[1] - z[0])
    weights = np.empty(n_z, dtype=np.float64)
    for column in range(n_z):
        basis = np.zeros((n_z, 1), dtype=np.float64)
        basis[column, 0] = 1.0
        coefficients = (h @ basis).reshape(4, n_z - 1, 1)
        weights[column] = np.sum(poly_int0(coefficients[0], coefficients[1], coefficients[2], coefficients[3], z[:-1, None], hz), axis=0)[0]
    weights.setflags(write=False)
    return weights


def xi_integral_batched_pairs_from_b_cache_direct_py(rho1, z_point_t, nb, b1,
                                                      jj_index, j_index_tot,
                                                      ll_index, l_index_tot,
                                                      b_cache, integration_operator):
    """
    xi_integral_batched_pairs_from_b_cache_direct_py
    
    PURPOSE:
        Evaluate pair-batched Xi using direct fixed-grid integrated responses.
    
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
    weights = np.asarray(integration_operator, dtype=np.float64)
    if weights.shape != (len(z_point_t),):
        raise ValueError("Invalid fixed-grid integration operator shape.")
    xi = np.zeros((nb, nb, nb, nb), dtype=complex)
    pair_order = [(jj, j) for jj in jj_index for j in j_index_tot[jj]]
    if not pair_order:
        return xi
    pair_jj = np.asarray([pair[0] for pair in pair_order], dtype=int)
    pair_j = np.asarray([pair[1] for pair in pair_order], dtype=int)
    rho1_pairs = np.conj(rho1[pair_jj, pair_j, :]).T
    for ll in ll_index:
        for l in l_index_tot[ll]:
            b_small, b_large = b_cache[(ll, l)]
            xi[pair_jj, pair_j, ll, l] = weights @ (rho1_pairs * b_small[:, None]) + weights @ (rho1_pairs * b_large[:, None])
    return xi



def build_fixed_grid_b_operators_py(G, kex, z_point_t):
    """
    build_fixed_grid_b_operators_py
    
    PURPOSE:
        Build exact fixed-(G,z) rho2-to-b response operators from reference b code.
    
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
    n_z = len(z_point_t)
    small = np.empty((n_z, n_z), dtype=np.complex128)
    large = np.empty((n_z, n_z), dtype=np.complex128)
    index = [[0]]
    for column in range(n_z):
        rho2 = np.zeros((1, 1, n_z), dtype=np.float64)
        rho2[0, 0, column] = 1.0
        b_small, b_large = build_b_cache_py(rho2, G, kex, z_point_t, [0], index)[(0, 0)]
        small[:, column] = b_small
        large[:, column] = b_large
    small.setflags(write=False)
    large.setflags(write=False)
    return small, large


def build_b_cache_operator_py(rho2, ll_index, l_index_tot, small_operator, large_operator):
    """
    build_b_cache_operator_py
    
    PURPOSE:
        Build b cache by applying fixed-(G,z) direct operators to ordered rho2 columns.
    
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
    pairs = [(ll, l) for ll in ll_index for l in l_index_tot[ll]]
    if not pairs:
        return {}
    jj = np.asarray([pair[0] for pair in pairs], dtype=int)
    j = np.asarray([pair[1] for pair in pairs], dtype=int)
    signals = rho2[jj, j, :].T
    small = np.asarray(small_operator, dtype=np.complex128) @ signals
    large = np.asarray(large_operator, dtype=np.complex128) @ signals
    return {pair: (small[:, column], large[:, column]) for column, pair in enumerate(pairs)}
