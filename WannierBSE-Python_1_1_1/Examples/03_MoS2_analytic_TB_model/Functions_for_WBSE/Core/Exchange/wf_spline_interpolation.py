"""
wf_spline_interpolation

PURPOSE:
    Cubic-spline coefficients for truncated Wannier-pair products.

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


def _kex_uv(kex, a1, a2):
    """
    _kex_uv
    
    PURPOSE:
        Kex uv.
    
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
    a1x, a1y = a1[:2]
    a2x, a2y = a2[:2]
    beta = (-a2x * a1y) + (a1x * a2y)
    return ((kex[0] * a1x * a2y) + (kex[1] * a1y * a2y)) / beta, ((kex[0] * a2x * a1y) + (kex[1] * a1y * a2y)) / beta


def wf_spline_interpolation_py(Wij_z, kex, a1, a2, u_2d, u_point_t, v_point_t, kex_tol):
    """
    wf_spline_interpolation_py
    
    PURPOSE:
        Wf spline interpolation py.
    
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
    nu = len(u_point_t)
    nv = len(v_point_t)
    kex_uu, _ = _kex_uv(np.asarray(kex), np.asarray(a1), np.asarray(a2))
    kexr_u = np.asarray(u_2d) * kex_uu if abs(kex_uu) < kex_tol else np.zeros_like(u_2d, dtype=float)
    rw = np.asarray(Wij_z).reshape((nu, nv), order="F") * np.exp(-1j * kexr_u).reshape((nu, nv), order="F")
    spl = CubicSpline(u_point_t, rw, axis=0, bc_type="not-a-knot")
    c = np.moveaxis(spl.c, 0, -1)  # (nu-1, nv, 4), order c3,c2,c1,c0
    return c[:, :, 0], c[:, :, 1], c[:, :, 2], c[:, :, 3]
