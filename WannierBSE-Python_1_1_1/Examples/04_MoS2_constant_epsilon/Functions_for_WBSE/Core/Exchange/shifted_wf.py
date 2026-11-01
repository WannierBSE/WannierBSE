"""
shifted_wf

PURPOSE:
    Exact direct-index extraction equivalent to the legacy Fortran-order roll.

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


def roll_source_indices(tin_all, dims, shift):
    """
    roll_source_indices
    
    PURPOSE:
        Return source flat indices for np.roll(array, shift)[tin_all].
    
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
    n0, n1, n2 = (int(x) for x in dims)
    s0, s1, s2 = (int(x) for x in shift)
    tin = np.asarray(tin_all, dtype=np.int64)
    i0 = tin % n0
    i1 = (tin // n0) % n1
    i2 = tin // (n0 * n1)
    src0 = (i0 - s0) % n0
    src1 = (i1 - s1) % n1
    src2 = (i2 - s2) % n2
    return (src0 + n0 * src1 + n0 * n1 * src2).astype(np.int64, copy=False)


def shifted_truncated_matrix_direct(W, source_indices):
    """
    shifted_truncated_matrix_direct
    
    PURPOSE:
        Extract all shifted truncated Wannier columns without a full-grid roll.
    
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
    values = np.asarray(W)
    if values.ndim != 2:
        raise ValueError("W must have shape (full_grid_points, nb)")
    return values[np.asarray(source_indices, dtype=np.int64), :]
