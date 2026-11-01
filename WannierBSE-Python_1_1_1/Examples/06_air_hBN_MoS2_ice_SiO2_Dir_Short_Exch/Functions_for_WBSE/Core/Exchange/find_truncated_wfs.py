"""
find_truncated_wfs

PURPOSE:
    Wrapper matching MATLAB find_truncated_wfs.

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

from .find_wfs_box import find_wfs_box_py


def find_truncated_wfs_py(W_up, W_down, r, u, v, delta_v, na1, z_point, tol_probability):
    """
    find_truncated_wfs_py
    
    PURPOSE:
        Find truncated wfs py.
    
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
    result = find_wfs_box_py(tol_probability, W_up, W_down, r, u, v, r[:, 2], delta_v, na1, z_point)
    W_up_t, W_down_t, tin_all, r_t, u_t, v_t, z_t, u_point_t, v_point_t, z_point_t, nu_t, nv_t, nz_t = result
    return (W_up_t, W_down_t, tin_all, r_t, u_t, v_t, z_t, u_point_t, v_point_t, z_point_t,
            nu_t, nv_t, nz_t, len(u_point_t), len(v_point_t), 1, nz_t)
