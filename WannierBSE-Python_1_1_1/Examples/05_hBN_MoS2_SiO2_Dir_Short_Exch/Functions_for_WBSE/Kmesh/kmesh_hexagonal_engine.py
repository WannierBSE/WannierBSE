"""
kmesh_hexagonal_engine

PURPOSE:
    Dual-resolution hexagonal kmesh generator used by WBSE.

DESCRIPTION:
    This is a direct Python translation of the MATLAB kmesh_hexagonal_engine.m
    logic, including deterministic topology metadata for future TB generation.

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

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class KmeshTopology:
    """
    KmeshTopology
    
    PURPOSE:
        Configuration or data container used by kmesh_hexagonal_engine.
    
    DESCRIPTION:
        The fields define the data contract for the associated workflow.
    
    INPUT ARGUMENTS:
        See the annotated signature and detailed description.
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    raw_index: int
    final_k_index: int
    sorted_output_index: int
    region_type: str
    region_id: str
    sector_id: int
    template_id: str
    patch_id: str
    mesh_group: str
    triangle_id: int
    local_i: int
    local_j: int
    triangle_orientation: str
    vertices: np.ndarray


def _get_hex_orientation(a1: np.ndarray, a2: np.ndarray) -> str:
    """
    _get_hex_orientation
    
    PURPOSE:
        Get hex orientation.
    
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
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    a = np.linalg.norm(a1)
    tol = 1e-2
    angle = np.degrees(np.arccos(np.dot(a1, a2) / (a * np.linalg.norm(a2))))
    if abs(a - np.linalg.norm(a2)) > tol or not np.any(np.abs(angle - np.array([60.0, 120.0])) < tol):
        return "Not Hexagonal"

    for vec in (a1, a2, a1 + a2, a1 - a2):
        if abs(vec[0]) < tol and abs(abs(vec[1]) - a) < tol:
            return "Horizontal"
        if abs(vec[1]) < tol and abs(abs(vec[0]) - a) < tol:
            return "Vertical"
    return "Neither"


def _make_topology(
    region_type: str,
    template_id: str,
    patch_id: str,
    mesh_group: str,
    local_i: int,
    local_j: int,
    orientation: str,
    triangle_id: int,
    vertices: np.ndarray,
) -> KmeshTopology:
    """
    _make_topology
    
    PURPOSE:
        Make topology.
    
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
    return KmeshTopology(
        raw_index=-1,
        final_k_index=-1,
        sorted_output_index=-1,
        region_type=region_type,
        region_id="",
        sector_id=-1,
        template_id=template_id,
        patch_id=patch_id,
        mesh_group=mesh_group,
        triangle_id=triangle_id,
        local_i=local_i,
        local_j=local_j,
        triangle_orientation=orientation,
        vertices=np.asarray(vertices, dtype=int),
    )


def _keep_triangle(mode: str, i: int, j: int, limit: int | None, is_down: bool) -> bool:
    """
    _keep_triangle
    
    PURPOSE:
        Keep triangle.
    
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
    if mode == "all":
        return True
    if limit is None:
        return False
    if mode == "lower":
        return i + int(is_down) < limit
    if mode == "upper":
        if is_down:
            return (i + 1 >= limit) and (j <= limit - 1) and ((i - j) <= limit - 1)
        return i >= limit and not (j > limit - 1 or (i - j) > limit - 1)
    return False


def _triangular_kmesh(
    n_start: int,
    n_end: int,
    u: np.ndarray,
    v: np.ndarray,
    shift: np.ndarray,
    limit: int | None,
    mode: str,
    region_type: str,
    template_id: str,
    patch_id: str,
    mesh_group: str,
) -> tuple[np.ndarray, list[KmeshTopology]]:
    """
    _triangular_kmesh
    
    PURPOSE:
        Triangular kmesh.
    
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
    k_list: list[np.ndarray] = []
    mesh_lines: list[KmeshTopology] = []
    triangle_id = 0

    for i in range(n_start, n_end + 1):
        for j in range(0, i + 1):
            p1 = (i - j) * u + j * v + shift
            p2 = (i - j + 1) * u + j * v + shift
            p3 = (i - j) * u + (j + 1) * v + shift
            c_up = (p1 + p2 + p3) / 3.0
            vertices_up = np.array([[i - j, j], [i - j + 1, j], [i - j, j + 1]], dtype=int)

            if _keep_triangle(mode, i, j, limit, False):
                triangle_id += 1
                k_list.append(c_up)
                mesh_lines.append(
                    _make_topology(
                        region_type, template_id, patch_id, mesh_group, i, j, "up", triangle_id, vertices_up
                    )
                )

            if i + 1 <= n_end:
                pd1 = (i - j + 1) * u + j * v + shift
                pd2 = (i - j) * u + (j + 1) * v + shift
                pd3 = (i - j + 1) * u + (j + 1) * v + shift
                c_down = (pd1 + pd2 + pd3) / 3.0
                vertices_down = np.array(
                    [[i - j + 1, j], [i - j, j + 1], [i - j + 1, j + 1]], dtype=int
                )

                if _keep_triangle(mode, i, j, limit, True):
                    triangle_id += 1
                    k_list.append(c_down)
                    mesh_lines.append(
                        _make_topology(
                            region_type,
                            template_id,
                            patch_id,
                            mesh_group,
                            i,
                            j,
                            "down",
                            triangle_id,
                            vertices_down,
                        )
                    )

    if k_list:
        return np.vstack(k_list), mesh_lines
    return np.zeros((0, 2), dtype=float), mesh_lines


def _edge_key(meta: KmeshTopology, vertices: np.ndarray) -> str:
    """
    _edge_key
    
    PURPOSE:
        Edge key.
    
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
    vertices = np.asarray(vertices, dtype=int)
    vertices = vertices[np.lexsort((vertices[:, 1], vertices[:, 0]))]
    return (
        f"{meta.sector_id}|{meta.template_id}|{meta.mesh_group}|"
        f"{vertices[0, 0]},{vertices[0, 1]};{vertices[1, 0]},{vertices[1, 1]}"
    )


def _build_generator_neighbors(topology: list[KmeshTopology]) -> np.ndarray:
    """
    _build_generator_neighbors
    
    PURPOSE:
        Build generator neighbors.
    
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
    edge_map: dict[str, list[int]] = {}
    pairs: list[tuple[int, int]] = []
    for idx, meta in enumerate(topology, start=1):
        vertices = meta.vertices
        edge_vertices = (vertices[[0, 1], :], vertices[[1, 2], :], vertices[[0, 2], :])
        for edge in edge_vertices:
            key = _edge_key(meta, edge)
            if key in edge_map:
                for previous in edge_map[key]:
                    pairs.append(tuple(sorted((previous, idx))))
                edge_map[key].append(idx)
            else:
                edge_map[key] = [idx]

    if not pairs:
        return np.zeros((0, 2), dtype=int)
    return np.array(sorted(set(pairs)), dtype=int)


def kmesh_hexagonal_engine_py(
    nk_coarse: int,
    nk_dense: int,
    dense_range: int,
    a1: np.ndarray,
    a2: np.ndarray,
    a_const: float,
) -> tuple[np.ndarray, int, int, str, list[KmeshTopology], np.ndarray]:
    """
    kmesh_hexagonal_engine_py
    
    PURPOSE:
        Kmesh hexagonal engine py.
    
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
    orientation = _get_hex_orientation(a1, a2)

    if orientation == "Horizontal":
        u_basis = np.array([2 / np.sqrt(3), 2 / 3], dtype=float)
        v_basis = np.array([2 / np.sqrt(3), -2 / 3], dtype=float)
        w_basis = np.array([0, -4 / 3], dtype=float)
    elif orientation == "Vertical":
        u_basis = np.array([2 / 3, 2 / np.sqrt(3)], dtype=float)
        v_basis = np.array([-2 / 3, 2 / np.sqrt(3)], dtype=float)
        w_basis = np.array([-4 / 3, 0], dtype=float)
    else:
        raise ValueError(f"Unsupported hexagonal lattice orientation: {orientation}")

    u = u_basis / nk_coarse
    v = v_basis / nk_coarse
    w = w_basis / nk_coarse

    u_d = (u_basis * (dense_range / nk_coarse)) / nk_dense
    v_d = (v_basis * (dense_range / nk_coarse)) / nk_dense
    w_d = (w_basis * (dense_range / nk_coarse)) / nk_dense

    shift_u = u_basis * (1 - dense_range / nk_coarse)
    shift_v = v_basis * (1 - dense_range / nk_coarse)
    shift_w = w_basis * (1 - dense_range / nk_coarse)
    limit_lower = nk_coarse - dense_range

    k_low1, m_low1 = _triangular_kmesh(0, nk_coarse - 1, u, v, np.zeros(2), limit_lower, "lower", "coarse", "UV", "coarse_lower", "coarse")
    k_up1, m_up1 = _triangular_kmesh(0, nk_coarse - 1, u, v, np.zeros(2), limit_lower, "upper", "coarse", "UV", "coarse_upper", "coarse")
    k_du1, m_du1 = _triangular_kmesh(0, nk_dense - 1, u_d, v_d, shift_u, None, "all", "dense", "UV", "dense_U", "dense_U")
    k_dv1, m_dv1 = _triangular_kmesh(0, nk_dense - 1, u_d, v_d, shift_v, None, "all", "dense", "UV", "dense_V", "dense_V")
    master_uv = np.vstack((k_low1, k_up1, k_du1, k_dv1))
    meta_uv = m_low1 + m_up1 + m_du1 + m_dv1

    k_low2, m_low2 = _triangular_kmesh(0, nk_coarse - 1, v, w, np.zeros(2), limit_lower, "lower", "coarse", "VW", "coarse_lower", "coarse")
    k_up2, m_up2 = _triangular_kmesh(0, nk_coarse - 1, v, w, np.zeros(2), limit_lower, "upper", "coarse", "VW", "coarse_upper", "coarse")
    k_dv2, m_dv2 = _triangular_kmesh(0, nk_dense - 1, v_d, w_d, shift_v, None, "all", "dense", "VW", "dense_V", "dense_V")
    k_dw2, m_dw2 = _triangular_kmesh(0, nk_dense - 1, v_d, w_d, shift_w, None, "all", "dense", "VW", "dense_W", "dense_W")
    master_vw = np.vstack((k_low2, k_up2, k_dv2, k_dw2))
    meta_vw = m_low2 + m_up2 + m_dv2 + m_dw2

    if orientation == "Vertical":
        mirrors = (
            np.array([1, 1]),
            np.array([1, 1]),
            np.array([1, -1]),
            np.array([1, -1]),
            np.array([-1, -1]),
            np.array([-1, 1]),
        )
    else:
        mirrors = (
            np.array([1, 1]),
            np.array([1, 1]),
            np.array([-1, 1]),
            np.array([-1, 1]),
            np.array([-1, -1]),
            np.array([1, -1]),
        )

    sectors = (master_uv, master_vw, master_vw, master_uv, master_vw, master_vw)
    sector_meta = (meta_uv, meta_vw, meta_vw, meta_uv, meta_vw, meta_vw)

    final_parts: list[np.ndarray] = []
    topology: list[KmeshTopology] = []
    raw_index = 0
    for sector_id, (sector_k, metas, mirror) in enumerate(zip(sectors, sector_meta, mirrors), start=1):
        final_parts.append(sector_k * mirror)
        for meta in metas:
            raw_index += 1
            topology.append(
                replace(
                    meta,
                    raw_index=raw_index,
                    sector_id=sector_id,
                    region_id=f"S{sector_id}_{meta.template_id}_{meta.patch_id}",
                )
            )

    final_kmesh = np.vstack(final_parts)

    num_coarse_uv = k_low1.shape[0] + k_up1.shape[0]
    num_dense_uv = k_du1.shape[0] + k_dv1.shape[0]
    num_coarse_vw = k_low2.shape[0] + k_up2.shape[0]
    num_dense_vw = k_dv2.shape[0] + k_dw2.shape[0]
    total_coarse = (num_coarse_uv * 2) + (num_coarse_vw * 4)
    total_dense = (num_dense_uv * 2) + (num_dense_vw * 4)

    neighbor_pairs = _build_generator_neighbors(topology)
    _ = a_const
    return final_kmesh, total_coarse, total_dense, orientation, topology, neighbor_pairs
