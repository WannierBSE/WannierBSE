"""
reset_internal_kmesh_tb_cache

PURPOSE:
    Reset internally generated kmesh/TB cache files.

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

import shutil
from pathlib import Path


def reset_internal_kmesh_tb_cache_py(
    kmesh_cache_file: str | Path,
    tb_cache_path: str | Path,
    temp_topology_path: str | Path,
    remove_kmesh: bool,
) -> None:
    """
    reset_internal_kmesh_tb_cache_py
    
    PURPOSE:
        Reset internal kmesh tb cache py.
    
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
    kmesh_cache_file = Path(kmesh_cache_file)
    tb_cache_path = Path(tb_cache_path)
    temp_topology_path = Path(temp_topology_path)

    if remove_kmesh and kmesh_cache_file.is_file():
        kmesh_cache_file.unlink()
    if tb_cache_path.is_dir():
        shutil.rmtree(tb_cache_path)
    tb_cache_path.mkdir(parents=True, exist_ok=True)
    if remove_kmesh and temp_topology_path.is_dir():
        shutil.rmtree(temp_topology_path)
