"""
resolve_data_path

PURPOSE:
    Resolve WBSE data files against User_input and Precomputed_data.

DESCRIPTION:
    This mirrors the MATLAB resolve_data_path.m contract:
    User_input/<name> wins, then Precomputed_data/<stem>_WBSE<suffix>,
    otherwise the caller must generate the file.

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


def resolve_data_path_py(
    file_name: str | Path,
    user_input_path: str | Path,
    cache_path: str | Path,
) -> tuple[Path | None, str]:
    """
    resolve_data_path_py
    
    PURPOSE:
        Resolve data path py.
    
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
    requested = Path(file_name)
    user_input_path = Path(user_input_path)
    cache_path = Path(cache_path)

    user_file = user_input_path / requested
    if user_file.is_file():
        return user_file, "LOAD "

    cache_file = cache_path / f"{requested.stem}_WBSE{requested.suffix}"
    if cache_file.is_file():
        return cache_file, "CACHE"

    return None, "GEN  "
