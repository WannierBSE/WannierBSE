"""
g_vector_v2

PURPOSE:
    MATLAB g_vector_v2 translation.

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


def g_vector_v2_py(glim, b1, b2):
    """
    g_vector_v2_py
    
    PURPOSE:
        G vector v2 py.
    
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
    glim = float(glim)
    b1 = np.asarray(b1, dtype=float).reshape(-1)[:2]
    b2 = np.asarray(b2, dtype=float).reshape(-1)[:2]
    if glim == 0:
        return np.array([[0.0, 0.0]])
    vals = np.arange(-int(np.ceil(glim)), int(np.ceil(glim)) + 1)
    comp = []
    for j in vals:
        for i in vals:
            comp.append([i, j])
    comp = np.asarray(comp, dtype=float)
    g = comp[:, [0]] * b1[None, :] + comp[:, [1]] * b2[None, :]
    gmax = glim * np.linalg.norm(b1)
    return g[np.linalg.norm(g, axis=1) <= gmax * 1.005]
