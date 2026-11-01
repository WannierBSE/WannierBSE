"""
spline_integral_z

PURPOSE:
    Analytic cubic-spline integral kernels for z exchange.

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
from .spline_integral_inplane import poly_int0


def poly_int_exp_real(c3, c2, c1, c0, G, y0, h, yc):
    """
    poly_int_exp_real
    
    PURPOSE:
        Poly int exp real.
    
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
    G = np.asarray(G)
    if np.all(np.abs(G) < 1.0e-14):
        return poly_int0(c3, c2, c1, c0, y0, h)
    a = -G
    exp0 = np.exp(-G * (y0 - yc))
    exp_h = np.exp(a * h)
    i0 = (exp_h - 1) / a
    i1 = exp_h * (h / a - 1 / a**2) + 1 / a**2
    i2 = exp_h * (h**2 / a - 2 * h / a**2 + 2 / a**3) - 2 / a**3
    i3 = exp_h * (h**3 / a - 3 * h**2 / a**2 + 6 * h / a**3 - 6 / a**4) + 6 / a**4
    return exp0 * (c3 * i3 + c2 * i2 + c1 * i1 + c0 * i0)
