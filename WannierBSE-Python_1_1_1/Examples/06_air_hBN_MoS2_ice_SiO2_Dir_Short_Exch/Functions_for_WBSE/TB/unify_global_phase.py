"""
unify_global_phase

PURPOSE:
    Global phase convention for WBSE tight-binding output.

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


def unify_global_phase_py(vec: np.ndarray, tol: float = 1.0e-14) -> np.ndarray:
    """
    unify_global_phase_py
    
    PURPOSE:
        Fix a complex eigenvector phase using the phase of its coefficient sum.
    
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
    phase_sum = np.sum(vec)
    if abs(phase_sum) > tol:
        return vec * np.conj(phase_sum) / abs(phase_sum)
    return vec
