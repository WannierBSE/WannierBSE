"""
coordinate_transformation

PURPOSE:
    Oblique coordinate transform used by exchange spline integrals.

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


def coordinate_transformation_py(r, a1, a2, nz):
    """
    coordinate_transformation_py
    
    PURPOSE:
        Coordinate transformation py.
    
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
    r = np.asarray(r, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    a1x, a1y = a1[:2]
    a2x, a2y = a2[:2]
    u = x - (a2x / a2y) * y
    v = -x + (a1x / a1y) * y
    u_point = unique_tol(u, 1.0e-6)
    v_point = unique_tol(v, 1.0e-6)
    z_point = np.linspace(float(np.min(z)), float(np.max(z)), int(nz))
    hu = float(u_point[1] - u_point[0])
    hv = float(v_point[1] - v_point[0])
    hz = float((np.max(z) - np.min(z)) / (int(nz) - 1))
    beta = (-a2x * a1y) + (a1x * a2y)
    jac = ((a1x * a1y * (a2y**2)) - (a2x * (a1y**2) * a2y)) / (beta**2)
    return float(jac), hu, hv, hz, u, v, u_point, v_point, z_point
