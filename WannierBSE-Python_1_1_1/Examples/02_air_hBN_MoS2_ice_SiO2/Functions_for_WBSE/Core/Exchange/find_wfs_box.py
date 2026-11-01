"""
find_wfs_box

PURPOSE:
    Find a common truncated Wannier-function box.

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
from .array_utils import unique_tol


def find_wfs_box_py(tol_probability, W_up, W_down, r, u, v, z, delta_v, na1, z_point):
    """
    find_wfs_box_py
    
    PURPOSE:
        Find wfs box py.
    
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
    W_up = np.asarray(W_up)
    W_down = np.asarray(W_down)
    r = np.asarray(r, dtype=float)
    uv_box = np.zeros((6, W_up.shape[1]), dtype=float)
    for i in range(W_up.shape[1]):
        wi = np.conj(W_up[:, i]) * W_up[:, i] + np.conj(W_down[:, i]) * W_down[:, i]
        norm = delta_v * np.sum(wi)
        density = (np.conj(W_up[:, i] / np.sqrt(norm)) * (W_up[:, i] / np.sqrt(norm)) +
                   np.conj(W_down[:, i] / np.sqrt(norm)) * (W_down[:, i] / np.sqrt(norm)))
        order = np.argsort(np.real(density))[::-1]
        csum = delta_v * np.cumsum(np.real(density[order]))
        n = int(np.max(np.flatnonzero(csum <= tol_probability))) + 1
        uv_box[:, i] = [np.min(u[order[:n]]), np.max(u[order[:n]]), np.min(v[order[:n]]), np.max(v[order[:n]]), np.min(z[order[:n]]), np.max(z[order[:n]])]
    mins_maxs = [np.min(uv_box[0]), np.max(uv_box[1]), np.min(uv_box[2]), np.max(uv_box[3]), np.min(uv_box[4]), np.max(uv_box[5])]
    u_min, u_max, v_min, v_max, z_min, z_max = mins_maxs
    plane = int(na1) * int(na1)
    in_u = (u[:plane] >= u_min - 0.01) & (u[:plane] <= u_max + 0.01)
    in_v = (v[:plane] >= v_min - 0.01) & (v[:plane] <= v_max + 0.01)
    tin = np.flatnonzero(in_u & in_v)
    iz_min = int(np.flatnonzero(np.abs(z_point - z_min) < 1.0e-6)[0])
    iz_max = int(np.flatnonzero(np.abs(z_point - z_max) < 1.0e-6)[-1])
    tin_all = np.concatenate([tin + iz * plane for iz in range(iz_min, iz_max + 1)]).astype(int)
    r_t = r[tin_all]
    u_t, v_t, z_t = u[tin_all], v[tin_all], z[tin_all]
    return (W_up[tin_all], W_down[tin_all], tin_all, r_t, u_t, v_t, z_t,
            unique_tol(u_t, 1.0e-3), unique_tol(v_t, 1.0e-3), unique_tol(z_t, 1.0e-3),
            len(unique_tol(u_t, 1.0e-3)), len(unique_tol(v_t, 1.0e-3)), len(unique_tol(z_t, 1.0e-3)))
