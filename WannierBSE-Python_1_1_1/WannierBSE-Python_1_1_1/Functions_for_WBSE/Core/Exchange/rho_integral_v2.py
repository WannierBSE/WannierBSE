"""
rho_integral_v2

PURPOSE:
    Spline-based in-plane pair-density Fourier integral.

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
from .spline_integral_inplane import poly_int0, poly_int_exp_i
from .wf_spline_interpolation import _kex_uv


def rho_integral_v2_py(
    c3,
    c2,
    c1,
    c0,
    kex,
    a1,
    a2,
    u_2d,
    u_point_t,
    v_point_t,
    kex_tol,
    hu,
    hv,
    jac,
    G,
):
    """
    rho_integral_v2_py
    
    PURPOSE:
        Rho integral v2 py.
    
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
    kex_uu, kex_vv = _kex_uv(np.asarray(kex), np.asarray(a1), np.asarray(a2))
    guu, gvv = _kex_uv(np.asarray(G), np.asarray(a1), np.asarray(a2))
    if abs(kex_uu) < kex_tol and abs(kex_vv) > kex_tol:
        gvv += kex_vv
    elif abs(kex_uu) > kex_tol and abs(kex_vv) < kex_tol:
        guu += kex_uu
    elif not (abs(kex_uu) < kex_tol and abs(kex_vv) < kex_tol):
        guu += kex_uu
        gvv += kex_vv
    if abs(guu) < kex_tol:
        su = jac * np.sum(poly_int0(c3, c2, c1, c0, u_point_t[:-1, None], hu), axis=0)
    else:
        su = jac * np.sum(poly_int_exp_i(c3, c2, c1, c0, guu, u_point_t[:-1, None], hu), axis=0)
    if abs(kex_vv) < kex_tol:
        kexr_v = np.asarray(v_point_t) * kex_vv
    else:
        kexr_v = np.zeros(len(v_point_t), dtype=float)
    spl = CubicSpline(v_point_t, su * np.exp(-1j * kexr_v), bc_type="not-a-knot")
    c = spl.c.T
    if abs(gvv) < kex_tol:
        return np.sum(poly_int0(c[:, 0], c[:, 1], c[:, 2], c[:, 3], v_point_t[:-1], hv))
    return np.sum(poly_int_exp_i(c[:, 0], c[:, 1], c[:, 2], c[:, 3], gvv, v_point_t[:-1], hv))


def rho_integral_v2_batched_py(c3, c2, c1, c0, kex, a1, a2, u_2d, u_point_t,
                               v_point_t, kex_tol, hu, hv, jac, G_all):
    """
    rho_integral_v2_batched_py
    
    PURPOSE:
        Evaluate the scalar rho kernel for all G vectors in input order.
    
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
    G_all = np.asarray(G_all)
    n_g = len(G_all)
    kex_uu, kex_vv = _kex_uv(np.asarray(kex), np.asarray(a1), np.asarray(a2))

    guu = np.empty(n_g, dtype=float)
    gvv = np.empty(n_g, dtype=float)
    for gi, G in enumerate(G_all):
        guu[gi], gvv[gi] = _kex_uv(np.asarray(G), np.asarray(a1), np.asarray(a2))

    if abs(kex_uu) < kex_tol and abs(kex_vv) > kex_tol:
        gvv += kex_vv
    elif abs(kex_uu) > kex_tol and abs(kex_vv) < kex_tol:
        guu += kex_uu
    elif not (abs(kex_uu) < kex_tol and abs(kex_vv) < kex_tol):
        guu += kex_uu
        gvv += kex_vv

    u0 = np.asarray(u_point_t)[:-1, None]
    su = np.empty((len(v_point_t), n_g), dtype=complex)
    zero_guu = np.abs(guu) < kex_tol
    if np.any(zero_guu):
        su[:, zero_guu] = (
            jac * np.sum(poly_int0(c3, c2, c1, c0, u0, hu), axis=0)
        )[:, None]
    if np.any(~zero_guu):
        nonzero_guu = np.flatnonzero(~zero_guu)
        su[:, nonzero_guu] = jac * np.sum(
            poly_int_exp_i(
                c3[..., None], c2[..., None], c1[..., None], c0[..., None],
                guu[nonzero_guu][None, None, :], u0[..., None], hu,
            ),
            axis=0,
        )

    if abs(kex_vv) < kex_tol:
        kexr_v = np.asarray(v_point_t) * kex_vv
    else:
        kexr_v = np.zeros(len(v_point_t), dtype=float)
    coefficients = CubicSpline(
        v_point_t,
        su * np.exp(-1j * kexr_v)[:, None],
        axis=0,
        bc_type="not-a-knot",
    ).c

    rho = np.empty(n_g, dtype=complex)
    v0 = np.asarray(v_point_t)[:-1, None]
    c3v, c2v, c1v, c0v = coefficients[0], coefficients[1], coefficients[2], coefficients[3]
    zero_gvv = np.abs(gvv) < kex_tol
    if np.any(zero_gvv):
        zero_indices = np.flatnonzero(zero_gvv)
        rho[zero_indices] = np.sum(
            poly_int0(c3v[:, zero_indices], c2v[:, zero_indices],
                      c1v[:, zero_indices], c0v[:, zero_indices], v0, hv),
            axis=0,
        )
    if np.any(~zero_gvv):
        nonzero_indices = np.flatnonzero(~zero_gvv)
        rho[nonzero_indices] = np.sum(
            poly_int_exp_i(c3v[:, nonzero_indices], c2v[:, nonzero_indices],
                           c1v[:, nonzero_indices], c0v[:, nonzero_indices],
                           gvv[nonzero_indices][None, :], v0, hv),
            axis=0,
        )
    return rho
