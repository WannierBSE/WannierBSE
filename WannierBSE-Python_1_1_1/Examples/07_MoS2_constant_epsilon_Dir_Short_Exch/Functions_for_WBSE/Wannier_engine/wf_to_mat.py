"""
wf_to_mat

PURPOSE:
    Wannier XSF spinor preprocessing for WannierBSE.

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

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from Functions_for_WBSE.Loaders.read_win import read_win_py
from Functions_for_WBSE.Wannier_engine.wf_centers import reorder_wf_centers_py
from Functions_for_WBSE.Wannier_engine.xsf_parser import XSFGrid, list_xsf_files, make_cartesian_grid, parse_xsf_header, read_xsf_values


# OPT-INSTRUMENTATION
def _opt_emit(phase: str, elapsed: float, scope: str, items: str, input_shape: str) -> None:
    """
    _opt_emit
    
    PURPOSE:
        Opt emit.
    
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
    return


@dataclass(frozen=True)
class WannierPaths:
    """
    WannierPaths
    
    PURPOSE:
        Configuration or data container used by wf_to_mat.
    
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
    root: Path
    user_input: Path
    xsf_root: Path
    parameters: Path
    precomputed: Path
    final_output: Path


@dataclass(frozen=True)
class NormalizationGeometry:
    """
    NormalizationGeometry
    
    PURPOSE:
        WF-independent geometry used by the spinor normalization integral.
    
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

    dims: tuple[int, int, int]
    u_point: np.ndarray
    v_point: np.ndarray
    z_point: np.ndarray
    jacobian: float
    u_bounds: tuple[float, float]
    v_bounds: tuple[float, float]
    z_bounds: tuple[float, float]
    w_u: np.ndarray
    w_v: np.ndarray
    w_z: np.ndarray


def init_wannier_paths_py(root_path: str | Path | None = None) -> WannierPaths:
    """
    init_wannier_paths_py
    
    PURPOSE:
        Init wannier paths py.
    
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
    root = Path(root_path) if root_path is not None else Path(__file__).resolve().parents[2]
    return WannierPaths(
        root=root,
        user_input=root / "User_input",
        xsf_root=root / "User_input" / "Wannier_functions_xsf",
        parameters=root / "Parameters",
        precomputed=root / "Precomputed_data",
        final_output=root / "Precomputed_data" / "Wannier_functions_WBSE",
    )


def _component_files(xsf_root: Path) -> dict[str, list[Path]]:
    """
    _component_files
    
    PURPOSE:
        Component files.
    
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
    folders = {name: xsf_root / name for name in ("up_real", "up_imag", "down_real", "down_imag")}
    files = {name: list_xsf_files(folder) for name, folder in folders.items()}
    counts = {name: len(items) for name, items in files.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"XSF component file counts differ: {counts}")
    names = [path.name for path in files["up_real"]]
    for comp, paths in files.items():
        if [path.name for path in paths] != names:
            raise ValueError(f"XSF filenames in {comp} do not match up_real ordering")
    return files


def inspect_wannier_xsf_py(root_path: str | Path | None = None) -> dict:
    """
    inspect_wannier_xsf_py
    
    PURPOSE:
        Inspect wannier xsf py.
    
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
    paths = init_wannier_paths_py(root_path)
    files = _component_files(paths.xsf_root)
    grid = parse_xsf_header(files["up_real"][0])
    counts = {name: len(items) for name, items in files.items()}
    sizes = {name: int(sum(path.stat().st_size for path in items)) for name, items in files.items()}
    first_sizes = {name: int(items[0].stat().st_size) for name, items in files.items()}
    return {
        "xsf_root": str(paths.xsf_root),
        "component_counts": counts,
        "component_sizes_bytes": sizes,
        "first_file_sizes_bytes": first_sizes,
        "grid_dims": list(grid.dims),
        "n_points": grid.n_points,
        "origin": grid.origin.tolist(),
        "a1": grid.a1.tolist(),
        "a2": grid.a2.tolist(),
        "a3": grid.a3.tolist(),
        "atom_count": len(grid.atoms),
        "atoms": [[symbol, coords.tolist()] for symbol, coords in grid.atoms],
    }


def _write_grid_info(path: Path, grid: XSFGrid) -> None:
    """
    _write_grid_info
    
    PURPOSE:
        Write grid info.
    
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
    arr = np.vstack([
        np.array(grid.dims, dtype=float),
        grid.origin,
        grid.a1,
        grid.a2,
        grid.a3,
    ])
    np.savetxt(path, arr, fmt="%16.14f")


def _write_atom_positions(path: Path, grid: XSFGrid) -> None:
    """
    _write_atom_positions
    
    PURPOSE:
        Write atom positions.
    
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
    rows = [grid.origin] + [coords for _, coords in grid.atoms]
    np.savetxt(path, np.vstack(rows), fmt="%16.14f")


def _compute_shift_indices(
    r: np.ndarray,
    grid: XSFGrid,
    primitive_a1: np.ndarray,
    primitive_a2: np.ndarray,
    primitive_a3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    _compute_shift_indices
    
    PURPOSE:
        Compute shift indices.
    
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
    na1, na2, na3 = grid.dims
    a1s, a2s = grid.a1, grid.a2
    nsc = int(np.ceil(np.linalg.norm(a1s) / np.linalg.norm(primitive_a1)))
    a1 = nsc * primitive_a1
    a2 = nsc * primitive_a2
    tol_shift = 0.05

    r2 = r.copy()
    plane = na1 * na2
    edge_y0 = np.arange(na1)
    edge_x0 = np.arange(0, plane, na1)
    for zi in range(na3):
        offset = zi * plane
        r2[offset + edge_y0, :2] += a2[:2]
        r2[offset + edge_x0, :2] += a1[:2]

    anchor = r[na1 + 1]
    ix = np.tile(np.arange(na1, dtype=float), na2)
    iy = np.repeat(np.arange(na2, dtype=float), na1)
    target = (ix[:, None] * (a1s / (na1 - 1))[None, :]) + (iy[:, None] * (a2s / (na2 - 1))[None, :]) + anchor
    tree = cKDTree(r2[:plane, :2])
    _, inplane_rearrange = tree.query(target[:, :2], k=1)

    r_shifted = np.empty_like(r2)
    r_shifted[:, 2] = r2[:, 2]
    for zi in range(na3):
        rows = np.arange(plane) + zi * plane
        r_shifted[rows, :2] = r2[inplane_rearrange + zi * plane, :2]

    z_wrap = np.flatnonzero(r_shifted[:, 2] < -0.02)
    if z_wrap.size:
        r_shifted[z_wrap, 2] += primitive_a3[2]
    keep = np.ones(r_shifted.shape[0], dtype=bool)
    keep[z_wrap] = False
    final_order = np.concatenate([np.flatnonzero(keep), z_wrap])
    r_final = r_shifted[final_order]
    return inplane_rearrange, final_order, r_final


def _apply_shift(
    values: np.ndarray,
    inplane_rearrange: np.ndarray,
    final_order: np.ndarray,
    dims: tuple[int, int, int],
) -> np.ndarray:
    """
    _apply_shift
    
    PURPOSE:
        Apply shift.
    
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
    na1, na2, na3 = dims
    plane = na1 * na2
    shifted = np.empty_like(values)
    for zi in range(na3):
        rows = np.arange(plane) + zi * plane
        shifted[rows] = values[inplane_rearrange + zi * plane]
    return shifted[final_order]


def _spline_integral_axis(values: np.ndarray, points: np.ndarray, axis: int) -> np.ndarray:
    """
    _spline_integral_axis
    
    PURPOSE:
        Spline integral axis.
    
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
    spline = CubicSpline(points, values, axis=axis, bc_type="not-a-knot", extrapolate=True)
    return spline.integrate(float(points[0]), float(points[-1]))


def _unique_tol(values: np.ndarray, tol: float = 1.0e-6) -> np.ndarray:
    """
    _unique_tol
    
    PURPOSE:
        Unique tol.
    
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
    sorted_values = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if sorted_values.size == 0:
        return sorted_values
    kept = [sorted_values[0]]
    for value in sorted_values[1:]:
        if abs(value - kept[-1]) > tol:
            kept.append(value)
    return np.array(kept, dtype=float)


def _normalization_integral(
    w_up: np.ndarray,
    w_down: np.ndarray,
    dims: tuple[int, int, int],
    r: np.ndarray,
    primitive_a1: np.ndarray,
    primitive_a2: np.ndarray,
) -> float:
    """
    _normalization_integral
    
    PURPOSE:
        Normalization integral.
    
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
    nu, nv, nz = dims
    density = (np.abs(w_up) ** 2 + np.abs(w_down) ** 2).reshape((nu, nv, nz), order="F")
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    a1x, a1y = primitive_a1[:2]
    a2x, a2y = primitive_a2[:2]
    u = x - (a2x / a2y) * y
    v = -x + (a1x / a1y) * y
    u_point = _unique_tol(u, 1.0e-6)
    v_point = _unique_tol(v, 1.0e-6)
    z_point = np.linspace(float(np.min(z)), float(np.max(z)), nz)
    if u_point.size != nu or v_point.size != nv:
        raise ValueError(f"Unexpected transformed grid sizes: u={u_point.size}/{nu}, v={v_point.size}/{nv}")
    beta = (-a2x * a1y) + (a1x * a2y)
    jac = ((a1x * a1y * (a2y**2)) - (a2x * (a1y**2) * a2y)) / (beta**2)
    integ_u = jac * _spline_integral_axis(density, u_point, axis=0)
    integ_v = _spline_integral_axis(integ_u, v_point, axis=0)
    integ_z = _spline_integral_axis(integ_v, z_point, axis=0)
    return float(np.real(integ_z))


def _build_spline_integration_weights(points: np.ndarray) -> np.ndarray:
    """
    _build_spline_integration_weights
    
    PURPOSE:
        Build fixed-grid CubicSpline integral responses.
    
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
    weights = np.empty(points.size, dtype=np.float64)
    for index in range(points.size):
        basis = np.zeros(points.size, dtype=np.float64)
        basis[index] = 1.0
        weights[index] = float(np.real(_spline_integral_axis(basis, points, axis=0)))
    weights.setflags(write=False)
    return weights


def build_normalization_geometry(
    dims: tuple[int, int, int],
    r: np.ndarray,
    primitive_a1: np.ndarray,
    primitive_a2: np.ndarray,
) -> NormalizationGeometry:
    """
    build_normalization_geometry
    
    PURPOSE:
        Build the shared normalization geometry for preprocessing.
    
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
    nu, nv, nz = dims
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    a1x, a1y = primitive_a1[:2]
    a2x, a2y = primitive_a2[:2]
    u = x - (a2x / a2y) * y
    v = -x + (a1x / a1y) * y
    u_point = _unique_tol(u, 1.0e-6)
    v_point = _unique_tol(v, 1.0e-6)
    z_point = np.linspace(float(np.min(z)), float(np.max(z)), nz)
    if u_point.size != nu or v_point.size != nv:
        raise ValueError(f"Unexpected transformed grid sizes: u={u_point.size}/{nu}, v={v_point.size}/{nv}")
    beta = (-a2x * a1y) + (a1x * a2y)
    jac = ((a1x * a1y * (a2y**2)) - (a2x * (a1y**2) * a2y)) / (beta**2)
    w_u = _build_spline_integration_weights(u_point)
    w_v = _build_spline_integration_weights(v_point)
    w_z = _build_spline_integration_weights(z_point)
    return NormalizationGeometry(
        dims=(nu, nv, nz),
        u_point=u_point,
        v_point=v_point,
        z_point=z_point,
        jacobian=float(jac),
        u_bounds=(float(u_point[0]), float(u_point[-1])),
        v_bounds=(float(v_point[0]), float(v_point[-1])),
        z_bounds=(float(z_point[0]), float(z_point[-1])),
        w_u=w_u,
        w_v=w_v,
        w_z=w_z,
    )


def _normalization_integral_with_geometry(
    w_up: np.ndarray,
    w_down: np.ndarray,
    geometry: NormalizationGeometry,
) -> float:
    """
    _normalization_integral_with_geometry
    
    PURPOSE:
        Evaluate one WF normalization using shared geometry and splines.
    
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
    density = (np.abs(w_up) ** 2 + np.abs(w_down) ** 2).reshape(geometry.dims, order="F")
    integ_u = geometry.jacobian * _spline_integral_axis(density, geometry.u_point, axis=0)
    integ_v = _spline_integral_axis(integ_u, geometry.v_point, axis=0)
    integ_z = _spline_integral_axis(integ_v, geometry.z_point, axis=0)
    return float(np.real(integ_z))


_normalization_integral_with_geometry_reference = _normalization_integral_with_geometry


def _normalization_integral_with_weights(
    w_up: np.ndarray,
    w_down: np.ndarray,
    geometry: NormalizationGeometry,
) -> float:
    """
    _normalization_integral_with_weights
    
    PURPOSE:
        Evaluate the fixed-grid normalization integral.
    
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
    density = (np.abs(w_up) ** 2 + np.abs(w_down) ** 2).reshape(geometry.dims, order="F")
    integral = geometry.jacobian * np.einsum(
        "i,j,k,ijk->", geometry.w_u, geometry.w_v, geometry.w_z, density, optimize=True
    )
    return float(np.real(integral))


def _matlab_complex_dtype() -> np.dtype:
    """
    _matlab_complex_dtype
    
    PURPOSE:
        Matlab complex dtype.
    
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
    return np.dtype([("real", "<f8"), ("imag", "<f8")])


def _as_matlab_complex_row(values: np.ndarray) -> np.ndarray:
    """
    _as_matlab_complex_row
    
    PURPOSE:
        As matlab complex row.
    
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
    out = np.empty(values.shape, dtype=_matlab_complex_dtype())
    out["real"] = values.real
    out["imag"] = values.imag
    return out


def cleanup_wannier_xsf_py(root_path: str | Path | None = None) -> None:
    """
    cleanup_wannier_xsf_py
    
    PURPOSE:
        Delete only wannier90_*.xsf files from the four expected XSF component folders.
    
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
    paths = init_wannier_paths_py(root_path)
    print("[INFO ] Wannier: Removing input XSF files after successful completion...")
    for name in ("up_real", "up_imag", "down_real", "down_imag"):
        component_dir = paths.xsf_root / name
        if not component_dir.is_dir():
            continue
        for path in component_dir.glob("wannier90_*.xsf"):
            path.unlink()


def generate_wannier_cache_py(
    root_path: str | Path | None = None,
    overwrite: bool = False,
    cleanup_xsf: bool | None = None,
) -> dict:
    # OPT-INSTRUMENTATION
    """
    generate_wannier_cache_py
    
    PURPOSE:
        Generate wannier cache py.
    
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
    t_opt_total = time.perf_counter()
    t_opt = time.perf_counter()
    paths = init_wannier_paths_py(root_path)
    files = _component_files(paths.xsf_root)
    _opt_emit("xsf_discovery", time.perf_counter() - t_opt, "exclusive", "88", "NA")
    t_opt = time.perf_counter()
    grid = parse_xsf_header(files["up_real"][0])
    _opt_emit("xsf_header_parse", time.perf_counter() - t_opt, "exclusive", "1", str(grid.dims))
    nb = len(files["up_real"])
    n_points = grid.n_points
    required = ["WF_centers_WBSE.txt", "WF_up.mat", "WF_down.mat", "r.mat", "GridInfo.txt", "WF_AtomPosition.txt"]
    t_opt = time.perf_counter()
    if paths.final_output.is_dir():
        existing = [paths.final_output / name for name in required if (paths.final_output / name).is_file()]
        missing = [name for name in required if not (paths.final_output / name).is_file()]
        if missing:
            print(
                "[CACHE] Wannier: incomplete cache detected "
                f"(missing {', '.join(missing)}). Clearing {paths.final_output} before rebuild...",
                flush=True,
            )
            shutil.rmtree(paths.final_output)
        elif not overwrite:
            raise FileExistsError(
                f"Wannier cache files already exist; pass overwrite=True to replace: {existing}"
            )
        else:
            shutil.rmtree(paths.final_output)
    paths.final_output.mkdir(parents=True, exist_ok=True)
    _opt_emit("cache_check", time.perf_counter() - t_opt, "exclusive", "1", "NA")

    primitive_a1, primitive_a2, primitive_a3, *_ = read_win_py(paths.user_input)
    # OPT-INSTRUMENTATION
    t_opt = time.perf_counter()
    r_raw = make_cartesian_grid(grid)
    _opt_emit("cartesian_grid_build", time.perf_counter() - t_opt, "exclusive", "1", str(r_raw.shape))
    t_opt = time.perf_counter()
    inplane_rearrange, final_order, r_final = _compute_shift_indices(r_raw, grid, primitive_a1, primitive_a2, primitive_a3)
    _opt_emit("shift_map_build", time.perf_counter() - t_opt, "exclusive", "1", str(r_final.shape))
    t_opt = time.perf_counter()
    normalization_geometry = build_normalization_geometry(grid.dims, r_final, primitive_a1, primitive_a2)
    _opt_emit("normalization_geometry", time.perf_counter() - t_opt, "exclusive", "1", str(grid.dims))
    t_opt = time.perf_counter()
    reorder_map = reorder_wf_centers_py(paths.parameters / "WF_centers.txt", paths.final_output / "WF_centers_WBSE.txt")
    _opt_emit("wf_center_processing", time.perf_counter() - t_opt, "exclusive", str(nb), "NA")

    # OPT-INSTRUMENTATION
    t_opt = time.perf_counter()
    with h5py.File(paths.final_output / "r.mat", "w") as h5:
        dset = h5.create_dataset("r", data=r_final.T, dtype=np.float64, compression=None)
        dset.attrs["MATLAB_class"] = np.bytes_("double")
    _write_grid_info(paths.final_output / "GridInfo.txt", grid)
    _write_atom_positions(paths.final_output / "WF_AtomPosition.txt", grid)
    _opt_emit("output_dataset_creation", time.perf_counter() - t_opt, "exclusive", "3", str(grid.dims))

    # OPT-INSTRUMENTATION
    t_opt = time.perf_counter()
    up_file = h5py.File(paths.final_output / "WF_up.mat", "w")
    down_file = h5py.File(paths.final_output / "WF_down.mat", "w")
    w_up_out = up_file.create_dataset("W_up", shape=(nb, n_points), dtype=_matlab_complex_dtype(), chunks=(1, min(n_points, 262144)), compression=None)
    w_down_out = down_file.create_dataset("W_down", shape=(nb, n_points), dtype=_matlab_complex_dtype(), chunks=(1, min(n_points, 262144)), compression=None)
    w_up_out.attrs["MATLAB_class"] = np.bytes_("double")
    w_down_out.attrs["MATLAB_class"] = np.bytes_("double")

    _opt_emit("output_dataset_creation", time.perf_counter() - t_opt, "exclusive", "2", str((nb, n_points)))
    norms_before = np.zeros(nb, dtype=float)
    elapsed_by_wf = []
    # OPT-INSTRUMENTATION
    opt_totals = {name: 0.0 for name in ("xsf_values_read", "spinor_assembly", "shift_apply", "normalization_integral", "normalization_scale", "basis_reorder", "cache_write", "unclassified_overhead")}
    start = time.perf_counter()
    try:
        for old_idx in range(nb):
            t0 = time.perf_counter()
            # OPT-INSTRUMENTATION
            t_phase = time.perf_counter()
            up_real = read_xsf_values(files["up_real"][old_idx], n_points)
            up_imag = read_xsf_values(files["up_imag"][old_idx], n_points)
            down_real = read_xsf_values(files["down_real"][old_idx], n_points)
            down_imag = read_xsf_values(files["down_imag"][old_idx], n_points)
            opt_totals["xsf_values_read"] += time.perf_counter() - t_phase
            t_phase = time.perf_counter()
            up = up_real + 1j * up_imag
            down = down_real + 1j * down_imag
            del up_real, up_imag, down_real, down_imag
            opt_totals["spinor_assembly"] += time.perf_counter() - t_phase
            t_phase = time.perf_counter()
            up = _apply_shift(up, inplane_rearrange, final_order, grid.dims)
            down = _apply_shift(down, inplane_rearrange, final_order, grid.dims)
            opt_totals["shift_apply"] += time.perf_counter() - t_phase
            t_phase = time.perf_counter()
            norm = np.sqrt(_normalization_integral_with_weights(up, down, normalization_geometry))
            opt_totals["normalization_integral"] += time.perf_counter() - t_phase
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError(f"Invalid WF norm for orbital {old_idx + 1}: {norm}")
            t_phase = time.perf_counter()
            up /= norm
            down /= norm
            opt_totals["normalization_scale"] += time.perf_counter() - t_phase
            t_phase = time.perf_counter()
            new_idx = int(reorder_map[old_idx])
            opt_totals["basis_reorder"] += time.perf_counter() - t_phase
            t_phase = time.perf_counter()
            w_up_out[new_idx, :] = _as_matlab_complex_row(up)
            w_down_out[new_idx, :] = _as_matlab_complex_row(down)
            opt_totals["cache_write"] += time.perf_counter() - t_phase
            norms_before[old_idx] = norm
            elapsed_by_wf.append(time.perf_counter() - t0)
            print(f"  -> WF {old_idx + 1:02d}/{nb:02d} normalized and written as column {new_idx + 1:02d} ({elapsed_by_wf[-1]:.2f} s)", flush=True)
    finally:
        up_file.close()
        down_file.close()

    # OPT-INSTRUMENTATION
    for phase, elapsed in opt_totals.items():
        _opt_emit(phase, elapsed, "exclusive", str(nb), str((nb, n_points)))
    _opt_emit("unclassified_overhead", max(0.0, time.perf_counter() - start - sum(opt_totals.values())), "exclusive", str(nb), str((nb, n_points)))
    metrics = {
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "nb": nb,
        "n_points": n_points,
        "grid_dims": list(grid.dims),
        "reorder_map_1_based": (reorder_map + 1).tolist(),
        "cleanup_xsf_requested": None if cleanup_xsf is None else bool(cleanup_xsf),
        "cleanup_xsf_effect": "cleanup is deferred until successful program completion",
        "norm_before_min": float(np.min(norms_before)),
        "norm_before_max": float(np.max(norms_before)),
        "norm_before_mean": float(np.mean(norms_before)),
        "elapsed_seconds": float(time.perf_counter() - start),
        "elapsed_by_wf_seconds": elapsed_by_wf,
        "outputs": [str(paths.final_output / name) for name in required],
    }
    _opt_emit("preprocessing_total", time.perf_counter() - t_opt_total, "inclusive", "1", "NA")
    return metrics


def main() -> int:
    """
    main
    
    PURPOSE:
        Main.
    
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
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cleanup-xsf", action="store_true")
    parser.add_argument("--keep-xsf", action="store_true")
    args = parser.parse_args()

    if args.inspect_only:
        print(json.dumps(inspect_wannier_xsf_py(args.root), indent=2))
    else:
        cleanup_xsf = True if args.cleanup_xsf else False if args.keep_xsf else None
        print(json.dumps(generate_wannier_cache_py(args.root, overwrite=args.overwrite, cleanup_xsf=cleanup_xsf), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
