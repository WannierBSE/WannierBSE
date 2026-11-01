"""
kmesh_for_BSE

PURPOSE:
    Generate, sort, and export the WBSE kmesh cache.

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

from dataclasses import replace
from pathlib import Path

import numpy as np

from Functions_for_WBSE.Kmesh.kmesh_hexagonal_engine import KmeshTopology, kmesh_hexagonal_engine_py
from Functions_for_WBSE.Kmesh.kmesh_sorter import kmesh_sorter_py
from Functions_for_WBSE.Loaders.read_params import read_params_py
from Functions_for_WBSE.Loaders.read_win import read_win_py


def _sort_kmesh_topology(
    raw_topology: list[KmeshTopology],
    raw_neighbor_pairs: np.ndarray,
    index_map: np.ndarray,
) -> tuple[list[KmeshTopology], np.ndarray]:
    """
    _sort_kmesh_topology
    
    PURPOSE:
        Sort kmesh topology.
    
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
    sorted_topology = [raw_topology[int(idx)] for idx in index_map]
    inverse_index_map = np.zeros(len(index_map), dtype=int)
    inverse_index_map[index_map] = np.arange(1, len(index_map) + 1)

    sorted_topology = [
        replace(meta, final_k_index=i, sorted_output_index=i)
        for i, meta in enumerate(sorted_topology, start=1)
    ]

    if raw_neighbor_pairs.size == 0:
        return sorted_topology, np.zeros((0, 2), dtype=int)
    sorted_pairs = inverse_index_map[raw_neighbor_pairs - 1]
    sorted_pairs = np.sort(sorted_pairs, axis=1)
    sorted_pairs = np.array(sorted({tuple(row) for row in sorted_pairs}), dtype=int)
    return sorted_topology, sorted_pairs


def _build_neighbor_columns(
    n_points: int,
    neighbor_pairs: np.ndarray,
    max_neighbors: int,
) -> np.ndarray:
    """
    _build_neighbor_columns
    
    PURPOSE:
        Build neighbor columns.
    
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
    adjacency = np.zeros((n_points, max_neighbors), dtype=int)
    counts = np.zeros(n_points, dtype=int)
    for i_raw, j_raw in neighbor_pairs:
        i = int(i_raw) - 1
        j = int(j_raw) - 1
        counts[i] += 1
        counts[j] += 1
        if counts[i] > max_neighbors or counts[j] > max_neighbors:
            raise ValueError(f"More than {max_neighbors} neighbors for a k-point.")
        adjacency[i, counts[i] - 1] = j + 1
        adjacency[j, counts[j] - 1] = i + 1
    return adjacency


def _write_kmesh_topology_sidecars(
    temp_path: Path,
    sorted_topology: list[KmeshTopology],
    sorted_neighbor_pairs: np.ndarray,
) -> None:
    """
    _write_kmesh_topology_sidecars
    
    PURPOSE:
        Write kmesh topology sidecars.
    
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
    temp_path.mkdir(parents=True, exist_ok=True)

    topology_path = temp_path / "kmesh_WBSE_topology.txt"
    with topology_path.with_suffix(".txt.tmp").open("w") as handle:
        handle.write(
            "# k_index region_type region_id sector_id template_id patch_id "
            "triangle_id local_i local_j triangle_orientation sorted_output_index\n"
        )
        for meta in sorted_topology:
            handle.write(
                f"{meta.final_k_index:8d} {meta.region_type:<6s} {meta.region_id:<18s} "
                f"{meta.sector_id:3d} {meta.template_id:<2s} {meta.patch_id:<13s} "
                f"{meta.triangle_id:6d} {meta.local_i:6d} {meta.local_j:6d} "
                f"{meta.triangle_orientation:<4s} {meta.sorted_output_index:8d}\n"
            )
    topology_path.with_suffix(".txt.tmp").replace(topology_path)

    neighbors_path = temp_path / "kmesh_WBSE_neighbors.txt"
    adjacency = _build_neighbor_columns(len(sorted_topology), sorted_neighbor_pairs, 6)
    with neighbors_path.with_suffix(".txt.tmp").open("w") as handle:
        handle.write("# k_index neighbor_1 neighbor_2 neighbor_3 neighbor_4 neighbor_5 neighbor_6\n")
        for k_index, row in enumerate(adjacency, start=1):
            handle.write(f"{k_index:8d} " + " ".join(f"{value:8d}" for value in row) + "\n")
    neighbors_path.with_suffix(".txt.tmp").replace(neighbors_path)


def kmesh_for_BSE_py(root_path: str | Path) -> Path:
    """
    kmesh_for_BSE_py
    
    PURPOSE:
        Kmesh for bse py.
    
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
    root_path = Path(root_path)
    output_path = root_path / "Precomputed_data" / "kmesh_WBSE.txt"
    if output_path.is_file():
        print(f"[INFO ] Existing k-mesh file detected: {output_path}")
        return output_path

    param_file = root_path / "Parameters" / "kmesh_control.txt"
    if not param_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {param_file}")

    params = read_params_py(param_file)
    nk_coarse = int(params["Nk_coarse"])
    nk_dense = int(params["Nk_dense"])
    dense_range = int(params["Dense_range"])
    if dense_range > nk_coarse / 2:
        raise ValueError("Dense_range must be <= Nk_coarse/2 to prevent point duplication.")

    a1, a2, _a3, a_const, _tau, _n_atoms = read_win_py(root_path / "User_input")
    raw_kmesh, total_coarse, total_dense, _orientation, raw_topology, raw_neighbor_pairs = kmesh_hexagonal_engine_py(
        nk_coarse, nk_dense, dense_range, a1, a2, a_const
    )

    sorted_kmesh, index_map, _header = kmesh_sorter_py(raw_kmesh, total_dense, total_coarse)
    sorted_topology, sorted_neighbor_pairs = _sort_kmesh_topology(raw_topology, raw_neighbor_pairs, index_map)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale_factor = np.pi / float(a_const)
    with output_path.open("w") as handle:
        handle.write(f"{dense_range:d}\n{nk_coarse:d}\n{nk_dense:d}\n{total_coarse:d}\n{total_dense:d}\n")
        for kx, ky in sorted_kmesh:
            handle.write(f"{kx * scale_factor:20.16f}    {ky * scale_factor:20.16f}     {0.0:20.16f}\n")

    _write_kmesh_topology_sidecars(root_path / "Precomputed_data" / "TB_data" / "temp", sorted_topology, sorted_neighbor_pairs)
    return output_path
