"""
rho_operator

PURPOSE:
    Reusable separable operator for exchange pair densities.

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

from dataclasses import dataclass
import time

import numpy as np

from .rho_integral_v2 import rho_integral_v2_batched_py
from .wf_spline_interpolation import wf_spline_interpolation_py


@dataclass(frozen=True)
class RhoOperatorFactors:
    """
    RhoOperatorFactors
    
    PURPOSE:
        Configuration or data container used by rho_operator.
    
    DESCRIPTION:
        The fields define the data contract for the associated workflow.
    
    INPUT ARGUMENTS:
        See the annotated signature and detailed description.
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    U: np.ndarray
    V: np.ndarray
    anchor: np.ndarray
    build_seconds: float
    responses: int


def _legacy_plane_response(W, G_all, kex, a1, a2, u_t_plane,
                           u_point_t, v_point_t, kex_tol, hu, hv, jac):
    """
    _legacy_plane_response
    
    PURPOSE:
        Legacy plane response.
    
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
    W = np.asarray(W, dtype=np.complex128)
    c3, c2, c1, c0 = wf_spline_interpolation_py(
        W.reshape(-1, order="F"), kex, a1, a2, u_t_plane,
        u_point_t, v_point_t, kex_tol,
    )
    return rho_integral_v2_batched_py(
        c3, c2, c1, c0, kex, a1, a2, u_t_plane, u_point_t,
        v_point_t, kex_tol, hu, hv, jac, G_all,
    ).astype(np.complex128, copy=False)


def rho_operator_build_factors_py(
    G_all, kex, a1, a2, u_t_plane, u_point_t, v_point_t,
    kex_tol, hu, hv, jac, anchor_tol=1.0e-12,
):
    """
    rho_operator_build_factors_py
    
    PURPOSE:
        Build U,V from the existing scalar spline/integration response.
    
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
    started = time.perf_counter()
    G_all = np.asarray(G_all, dtype=np.float64)
    kex = np.asarray(kex, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    a2 = np.asarray(a2, dtype=np.float64)
    u_t_plane = np.asarray(u_t_plane, dtype=np.float64)
    u_point_t = np.asarray(u_point_t, dtype=np.float64)
    v_point_t = np.asarray(v_point_t, dtype=np.float64)
    nu, nv, ng = len(u_point_t), len(v_point_t), len(G_all)
    e0_u = np.zeros(nu, dtype=np.complex128)
    e0_v = np.zeros(nv, dtype=np.complex128)
    e0_u[0] = 1.0
    e0_v[0] = 1.0
    A = np.empty((ng, nu), dtype=np.complex128)
    B = np.empty((ng, nv), dtype=np.complex128)
    for m in range(nu):
        x = np.zeros(nu, dtype=np.complex128)
        x[m] = 1.0
        A[:, m] = _legacy_plane_response(np.outer(x, e0_v), G_all, kex, a1, a2, u_t_plane, u_point_t, v_point_t, kex_tol, hu, hv, jac)
    for n in range(nv):
        if n == 0:
            B[:, n] = A[:, 0]
        else:
            y = np.zeros(nv, dtype=np.complex128)
            y[n] = 1.0
            B[:, n] = _legacy_plane_response(np.outer(e0_u, y), G_all, kex, a1, a2, u_t_plane, u_point_t, v_point_t, kex_tol, hu, hv, jac)
    anchor = A[:, 0].copy()
    min_anchor = float(np.min(np.abs(anchor)))
    if min_anchor <= anchor_tol:
        raise FloatingPointError(f"rho operator anchor too small: {min_anchor:.6e} <= {anchor_tol:.6e}")
    return RhoOperatorFactors(
        U=(A / anchor[:, None]).astype(np.complex128, copy=False),
        V=B.astype(np.complex128, copy=False), anchor=anchor,
        build_seconds=time.perf_counter() - started, responses=nu + nv - 1,
    )


def rho_operator_evaluate_planes_py(U, V, W_znm):
    """
    rho_operator_evaluate_planes_py
    
    PURPOSE:
        Evaluate rho[z,g] = sum_mn U[g,m] W[z,m,n] V[g,n].
    
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
    U = np.asarray(U, dtype=np.complex128)
    V = np.asarray(V, dtype=np.complex128)
    W_znm = np.asarray(W_znm, dtype=np.complex128)
    if U.ndim != 2 or V.ndim != 2 or W_znm.ndim != 3:
        raise ValueError("U, V, and W_znm have invalid ranks")
    if U.shape != V.shape or W_znm.shape[1:] != (U.shape[1], V.shape[1]):
        raise ValueError(f"incompatible shapes: U={U.shape}, V={V.shape}, W={W_znm.shape}")
    return np.einsum("gm,zmn,gn->zg", U, W_znm, V, optimize=True)
