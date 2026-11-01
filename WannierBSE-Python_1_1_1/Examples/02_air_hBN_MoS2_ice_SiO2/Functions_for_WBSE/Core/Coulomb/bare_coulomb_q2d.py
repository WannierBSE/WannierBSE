"""
bare_coulomb_q2d

PURPOSE:
    Bare strict-2D/finite-thickness Coulomb kernel.

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
import numpy as np

def bare_coulomb_q2d_py(q, d):
    """
    bare_coulomb_q2d_py
    
    PURPOSE:
        Bare coulomb q2d py.
    
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
    q = np.asarray(q, dtype=float)
    out = np.zeros_like(q)
    mask = q > 0.0
    if float(d) == 0.0:
        out[mask] = 2.0 * np.pi / q[mask]
    else:
        qp = q[mask]
        out[mask] = (1.0 - (1.0 - np.exp(-float(d) * qp)) / (float(d) * qp)) * (4.0 * np.pi / (float(d) * qp**2))
    return out
