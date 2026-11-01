"""
read_grid_info

PURPOSE:
    Read Stage 5 Wannier grid metadata for exchange generation.

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

from pathlib import Path
import numpy as np


def read_grid_info_py(wf_path, a1):
    """
    read_grid_info_py
    
    PURPOSE:
        Read grid info py.
    
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
    grid = np.loadtxt(Path(wf_path) / "GridInfo.txt")
    na1, na2, na3 = (int(round(x)) for x in grid[0, :3])
    a1s, a2s, a3s = grid[2], grid[3], grid[4]
    delta_v = float(np.dot(np.cross(a1s, a2s), a3s) / ((na1 - 1) * (na2 - 1) * (na3 - 1)))
    supercell_size = int(np.ceil(np.linalg.norm(a1s) / np.linalg.norm(a1)))
    shift_a1 = int(round(na1 / supercell_size))
    shift_a2 = int(round(na2 / supercell_size))
    shift_a3 = 0
    return na1, na2, na3, delta_v, supercell_size, shift_a1, shift_a2, shift_a3
