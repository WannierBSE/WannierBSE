"""
generate_g_vectors

PURPOSE:
    Generate nonzero short-range exchange reciprocal vectors.

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
from .g_vector_v2 import g_vector_v2_py


def generate_g_vectors_py(gmax, b1, b2):
    """
    generate_g_vectors_py
    
    PURPOSE:
        Generate g vectors py.
    
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
    g_all = g_vector_v2_py(gmax, b1, b2)
    keep = ~((np.abs(g_all[:, 0]) == 0) & (np.abs(g_all[:, 1]) == 0))
    return g_all[keep]
