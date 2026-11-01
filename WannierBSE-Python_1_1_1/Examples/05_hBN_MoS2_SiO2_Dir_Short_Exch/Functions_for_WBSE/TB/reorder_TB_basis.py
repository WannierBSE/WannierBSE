"""
reorder_TB_basis

PURPOSE:
    Basis reordering for WBSE tight-binding coefficients.

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


def default_reorder_map_py(nb: int) -> np.ndarray:
    """
    default_reorder_map_py
    
    PURPOSE:
        Return the legacy old-index to new-index permutation.
    
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
    if int(nb) == 22:
        return np.array(
            [1, 12, 2, 13, 3, 14, 4, 15, 5, 16, 6, 17, 7, 18, 8, 19, 9, 20, 10, 21, 11, 22],
            dtype=int,
        ) - 1
    return np.arange(int(nb), dtype=int)


def reorder_TB_basis_py(
    c_original: np.ndarray,
    reorder_map: np.ndarray | None = None,
) -> np.ndarray:
    """
    reorder_TB_basis_py
    
    PURPOSE:
        Reorder coefficient tensors from Wannier90 ordering to WBSE spin-block order.
    
    DESCRIPTION:
        N_kpt x NB x N_bands. ``reorder_map[old] = new`` uses 0-based indices.
    
    INPUT ARGUMENTS:
        See the annotated signature and detailed description.
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    c_original = np.asarray(c_original)
    if c_original.ndim != 3:
        raise ValueError("C_original must have shape N_kpt x NB x N_bands.")
    nb = c_original.shape[1]
    if reorder_map is None:
        reorder_map = default_reorder_map_py(nb)
    reorder_map = np.asarray(reorder_map, dtype=int).reshape(-1)
    if reorder_map.size != nb or not np.array_equal(np.sort(reorder_map), np.arange(nb)):
        raise ValueError("Basis reorder map must be a permutation of 0:NB-1.")

    c_reordered = np.zeros_like(c_original)
    for old, new in enumerate(reorder_map):
        c_reordered[:, new, :] = c_original[:, old, :]
    return c_reordered
