"""
kmesh_sorter

PURPOSE:
    Deterministic WBSE kmesh sorter.

DESCRIPTION:
    The sorter preserves the MATLAB convention: split by nearest-neighbor
    spacing into dense and coarse pools, then sort each pool by kx and ky.

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
from scipy.spatial.distance import cdist


def _write_sorted_kmesh_cache(kmesh_file: str | Path, header, k_fixed: np.ndarray) -> None:
    """
    _write_sorted_kmesh_cache
    
    PURPOSE:
        Write a sorted user k-mesh using the WBSE five-line header format.
    
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
    if header is None:
        raise ValueError("Cannot write sorted user kmesh cache without source header.")
    kmesh_file = Path(kmesh_file)
    kmesh_file.parent.mkdir(parents=True, exist_ok=True)
    h_vals = [int(round(value)) for value in header]
    with kmesh_file.open("w") as handle:
        for value in h_vals:
            handle.write(f"{value:d}\n")
        for kx, ky in k_fixed:
            handle.write(f"{kx:20.16f}    {ky:20.16f}     {0.0:20.16f}\n")


def _read_kmesh_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    _read_kmesh_file
    
    PURPOSE:
        Read kmesh file.
    
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
    with path.open("r") as handle:
        header = np.array([float(handle.readline().strip()) for _ in range(5)])
        data = np.loadtxt(handle, usecols=(0, 1))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_co = int(header[3])
    n_de = int(header[4])
    actual_rows = data.shape[0]
    if n_co + n_de != actual_rows:
        raise ValueError(
            f"K-mesh validation failed for {path}: header N_de+N_co={n_de + n_co}, "
            f"but file contains {actual_rows} points."
        )
    return header, data


def kmesh_sorter_py(
    input_data: str | Path | np.ndarray,
    n_de_in: int | None = None,
    n_co_in: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    kmesh_sorter_py
    
    PURPOSE:
        Kmesh sorter py.
    
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
    if isinstance(input_data, (str, Path)):
        header, k_raw = _read_kmesh_file(Path(input_data))
        n_de = int(header[4])
        n_co = int(header[3])
    else:
        header = None
        k_raw = np.asarray(input_data, dtype=float)[:, :2]
        if n_de_in is None or n_co_in is None:
            raise ValueError("n_de_in and n_co_in are required when sorting an in-memory kmesh.")
        n_de = int(n_de_in)
        n_co = int(n_co_in)
        if n_de + n_co != k_raw.shape[0]:
            raise ValueError(
                f"In-memory kmesh count mismatch: n_de+n_co={n_de + n_co}, rows={k_raw.shape[0]}."
            )

    dist_matrix = cdist(k_raw, k_raw)
    dist_matrix[dist_matrix == 0] = np.inf
    min_distances = np.min(dist_matrix, axis=1)
    global_idx = np.argsort(min_distances, kind="mergesort")

    dense_pool = global_idx[:n_de]
    coarse_pool = global_idx[n_de:]

    dense_local = np.lexsort((k_raw[dense_pool, 1], k_raw[dense_pool, 0]))
    coarse_local = np.lexsort((k_raw[coarse_pool, 1], k_raw[coarse_pool, 0]))

    index_map = np.concatenate((dense_pool[dense_local], coarse_pool[coarse_local]))
    k_sorted = k_raw[index_map, :]
    return k_sorted, index_map, header
