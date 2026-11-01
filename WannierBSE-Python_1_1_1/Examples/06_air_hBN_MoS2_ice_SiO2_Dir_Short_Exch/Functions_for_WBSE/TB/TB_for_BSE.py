"""
TB_for_BSE

PURPOSE:
    Generate WBSE tight-binding band files from Wannier90 HR data.

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

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
import time

import numpy as np
from joblib import Parallel, delayed

from Functions_for_WBSE.Loaders.read_params import read_params_py
from Functions_for_WBSE.TB.reorder_TB_basis import reorder_TB_basis_py
from Functions_for_WBSE.TB.unify_global_phase import unify_global_phase_py


@dataclass
class TBOptions:
    """
    TBOptions
    
    PURPOSE:
        Configuration or data container used by TB_for_BSE.
    
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
    output_nv: int
    output_nc: int
    vbm: int
    cbm: int
    npar_tb: int = 1
    spin_threshold: float = 0.3
    overlap_degeneracy_gap_ev: float = 1.0e-5
    exact_rotation_gap_ev: float = 1.0e-10
    tr_k_tolerance: float = 1.0e-8
    tr_energy_tolerance_ev: float = 1.0e-6
    tr_spin_tolerance: float = 0.10
    d3h_k_tolerance: float = 1.0e-8
    d3h_energy_tolerance_ev: float = 1.0e-6
    d3h_candidate_padding: int = 4
    d3h_max_energy_shift_ev: float = 0.02
    reciprocal: np.ndarray | None = None


@dataclass
class WFMetadata:
    """
    WFMetadata
    
    PURPOSE:
        Configuration or data container used by TB_for_BSE.
    
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
    index: np.ndarray
    atom: np.ndarray
    spinor: np.ndarray


@dataclass
class TBSelection:
    """
    TBSelection
    
    PURPOSE:
        Configuration or data container used by TB_for_BSE.
    
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
    v_idx: np.ndarray
    c_idx: np.ndarray
    v_sz: np.ndarray
    c_sz: np.ndarray
    v_target_spin: np.ndarray
    c_target_spin: np.ndarray
    spin_paired: np.ndarray
    spin_ambiguous: np.ndarray
    overlap_used: np.ndarray
    overlap_ref: np.ndarray
    v_overlap: np.ndarray
    c_overlap: np.ndarray
    v_overlap_margin: np.ndarray
    c_overlap_margin: np.ndarray
    v_gap: np.ndarray
    c_gap: np.ndarray
    full_spectrum_sz: np.ndarray
    tr_partner: np.ndarray | None = None
    tr_relabel_applied: np.ndarray | None = None


def _cache_bands_exist(tb_data_path: str | Path, nv: int, nc: int) -> bool:
    """
    _cache_bands_exist
    
    PURPOSE:
        Return whether all requested valence and conduction band files exist.
    
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
    tb_data_path = Path(tb_data_path)
    required = [f"v{i}_TB_WBSE.txt" for i in range(1, nv + 1)]
    required.extend(f"c{i}_TB_WBSE.txt" for i in range(1, nc + 1))
    return all((tb_data_path / name).is_file() for name in required)


def TB_for_BSE_py(context: dict | object) -> None:
    """
    TB_for_BSE_py
    
    PURPOSE:
        Python equivalent of MATLAB ``TB_for_BSE(context)``.
    
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
    tb_start = time.perf_counter()
    ctx = _validate_context(context)
    out_dir = Path(ctx["tbOutputDir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    main_params = read_params_py(ctx["controlFile"])
    tb_params = read_params_py(ctx["wtbControlFile"])
    opts = TBOptions(
        output_nv=int(main_params["Nv"]),
        output_nc=int(main_params["Nc"]),
        vbm=int(tb_params["VBM"]) - 1,
        cbm=int(tb_params["CBM"]) - 1,
        npar_tb=int(main_params.get("Npar_TB", 1)),
        spin_threshold=_read_optional_scalar(ctx["wtbControlFile"], "TB_spin_threshold", 0.3),
        overlap_degeneracy_gap_ev=_read_optional_scalar(
            ctx["wtbControlFile"], "TB_overlap_degeneracy_gap_eV", 1.0e-5
        ),
        exact_rotation_gap_ev=_read_optional_scalar(ctx["wtbControlFile"], "TB_exact_rotation_gap_eV", 1.0e-10),
        tr_k_tolerance=_read_optional_scalar(ctx["wtbControlFile"], "TB_time_reversal_k_tolerance", 1.0e-8),
        tr_energy_tolerance_ev=_read_optional_scalar(
            ctx["wtbControlFile"], "TB_time_reversal_energy_tolerance_eV", 1.0e-6
        ),
        tr_spin_tolerance=_read_optional_scalar(ctx["wtbControlFile"], "TB_time_reversal_spin_tolerance", 0.10),
        d3h_k_tolerance=_read_optional_scalar(ctx["wtbControlFile"], "TB_d3h_k_tolerance", 1.0e-8),
        d3h_energy_tolerance_ev=_read_optional_scalar(ctx["wtbControlFile"], "TB_d3h_energy_tolerance_eV", 1.0e-6),
        d3h_candidate_padding=int(round(_read_optional_scalar(ctx["wtbControlFile"], "TB_d3h_candidate_padding", 4))),
        d3h_max_energy_shift_ev=_read_optional_scalar(ctx["wtbControlFile"], "TB_d3h_max_energy_shift_eV", 0.02),
    )

    print(f"[LOAD ] TB: Reading WBSE k-mesh from {_tb_display_path(ctx['kmeshFile'])}")
    k = _read_wbse_kmesh(ctx["kmeshFile"])
    plv, lattice_source = _resolve_lattice_vectors(ctx)
    print(f"[LOAD ] TB: Reading {_tb_display_path(lattice_source)} and {_tb_display_path(ctx['wannierHrFile'])}")
    opts.reciprocal = 2.0 * np.pi * np.linalg.inv(plv).T

    nb, blvd, degn, hr = _read_wannier90_hr(ctx["wannierHrFile"])
    blvc = blvd @ plv
    wf = _read_wf_centers_spinor(ctx["wfCentersFile"])
    reorder_map = _spin_block_reorder_map(wf.spinor)
    _validate_inputs(k, nb, opts, reorder_map)

    e_data, c_data, diagonalization_timing = _diagonalize_all_kpoints(
        k, blvc, degn, hr, nb, opts.npar_tb
    )
    c_perm = np.transpose(c_data, (2, 0, 1))
    c_reordered = reorder_TB_basis_py(c_perm, reorder_map)
    c_data = np.transpose(c_reordered, (1, 2, 0))

    selection_start = time.perf_counter()
    spin_blocks = _spin_blocks_after_reorder(wf.spinor, reorder_map)
    c_data = _spin_adapt_exact_degenerate_clusters(e_data, c_data, spin_blocks, opts)
    selection = _select_output_window_by_band_index(e_data, c_data, opts, spin_blocks)
    selection = _enforce_conduction_spin_pairing(selection, e_data, opts)
    denominator = _read_mesh_denominator(ctx["kmeshFile"])
    assignment_cache: dict[tuple, np.ndarray] = {}
    selection = _enforce_d3h_energy_orbit_consistency_stage2e2(
        selection, e_data, k, opts, denominator, assignment_cache
    )
    selection = _refresh_selected_spins(c_data, selection, spin_blocks)
    # Build the exact indexed TR map once and pass it to both TR operations.
    tr_partner = _find_time_reversal_partners_stage2e2(k, opts, denominator)
    selection = _enforce_time_reversal_partner_states_stage2a(
        selection, e_data, c_data, k, opts, tr_partner
    )
    selection = _relabel_time_reversal_pairs_stage2a(selection, e_data, opts, tr_partner)
    c_data = _align_time_reversal_gauge(c_data, e_data, selection, opts)
    selection = _refresh_selected_spins(c_data, selection, spin_blocks)
    tr_diag = _time_reversal_diagnostics(e_data, selection, opts)
    if not tr_diag["passed"]:
        raise RuntimeError(f"Time-reversal energy validation failed during TB generation: {tr_diag['failure_reason']}.")

    write_start = time.perf_counter()
    _write_selected_band_files(out_dir, k, e_data, c_data, selection, opts)
    print(f"[DONE ] TB: Wrote WBSE band files to {_tb_display_path(out_dir)}")


def _context_get(context: dict | object, key: str):
    """
    _context_get
    
    PURPOSE:
        Context get.
    
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
    if isinstance(context, dict):
        return context.get(key)
    return getattr(context, key, None)


def _validate_context(context: dict | object) -> dict:
    """
    _validate_context
    
    PURPOSE:
        Validate context.
    
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
    required = ["kmeshFile", "wannierWinFile", "wannierHrFile", "wfCentersFile", "controlFile", "wtbControlFile", "tbOutputDir"]
    ctx = {key: _context_get(context, key) for key in required}
    structure = _context_get(context, "structureFile")
    ctx["structureFile"] = structure if structure else Path(ctx["wannierWinFile"]).with_name("structure.txt")
    exchange = _context_get(context, "exchangeInteraction")
    ctx["exchangeInteraction"] = True if exchange is None else bool(exchange)
    for key in ["kmeshFile", "wannierHrFile", "wfCentersFile", "controlFile", "wtbControlFile"]:
        path = ctx[key]
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"Required TB input not found: {path}")
    if ctx["exchangeInteraction"] and not Path(ctx["wannierWinFile"]).is_file():
        raise FileNotFoundError(f"Exchange interaction requires Wannier90 lattice input: {ctx['wannierWinFile']}")
    if not ctx["exchangeInteraction"] and not Path(ctx["wannierWinFile"]).is_file() and not Path(ctx["structureFile"]).is_file():
        raise FileNotFoundError(f"Required lattice input not found: {ctx['wannierWinFile']} or {ctx['structureFile']}")
    return ctx


def _read_optional_scalar(path: str | Path, key: str, fallback: float) -> float:
    """
    _read_optional_scalar
    
    PURPOSE:
        Read optional scalar.
    
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
    path = Path(path)
    if not path.is_file():
        return fallback
    for line in path.read_text().splitlines():
        stripped = line.split("%", 1)[0].strip()
        if not stripped or "=" not in stripped:
            continue
        lhs, rhs = stripped.split("=", 1)
        if lhs.strip() == key:
            return float(rhs.replace("D", "E").replace("d", "e").strip().rstrip(";"))
    return fallback


def _read_wbse_kmesh(path: str | Path) -> np.ndarray:
    """
    _read_wbse_kmesh
    
    PURPOSE:
        Read wbse kmesh.
    
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
    data = np.loadtxt(path, skiprows=5)
    data = np.atleast_2d(data)
    if data.shape[1] < 3:
        raise ValueError(f"Expected at least three k-coordinate columns in {path}.")
    return data[:, :3]


def _read_lattice_vectors_from_win(path: str | Path) -> np.ndarray:
    """
    _read_lattice_vectors_from_win
    
    PURPOSE:
        Read lattice vectors from win.
    
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
    lines = Path(path).read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if "begin unit_cell_cart" in line.lower()), None)
    end = next((i for i, line in enumerate(lines) if "end unit_cell_cart" in line.lower()), None)
    if start is None or end is None or end <= start:
        raise ValueError(f"Cannot find unit_cell_cart block in {path}.")
    rows = []
    for line in lines[start + 1 : end]:
        vals = np.fromstring(line, sep=" ")
        if vals.size >= 3:
            rows.append(vals[:3])
            if len(rows) == 3:
                return np.vstack(rows)
    raise ValueError(f"Cannot parse three lattice vectors from {path}.")


def _read_lattice_vectors_from_structure(path: str | Path) -> np.ndarray:
    """
    _read_lattice_vectors_from_structure
    
    PURPOSE:
        Read lattice vectors from structure.
    
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
    lines = Path(path).read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if "begin unit_cell_cart" in line.lower()), None)
    if start is None or start + 3 >= len(lines):
        raise ValueError(f"Cannot parse unit_cell_cart block in {path}.")
    rows = []
    for line in lines[start + 1 : start + 4]:
        vals = np.fromstring(line, sep=" ")
        if vals.size < 3:
            raise ValueError(f"Cannot parse lattice vector in {path}: {line}")
        rows.append(vals[:3])
    return np.vstack(rows)


def _resolve_lattice_vectors(ctx: dict) -> tuple[np.ndarray, Path]:
    """
    _resolve_lattice_vectors
    
    PURPOSE:
        Resolve lattice vectors.
    
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
    win = Path(ctx["wannierWinFile"])
    if win.is_file():
        return _read_lattice_vectors_from_win(win), win
    structure = Path(ctx["structureFile"])
    return _read_lattice_vectors_from_structure(structure), structure


def _read_wannier90_hr(path: str | Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    _read_wannier90_hr
    
    PURPOSE:
        Read wannier90 hr.
    
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
    with Path(path).open() as handle:
        handle.readline()
        nb = int(handle.readline().strip())
        n_blv = int(handle.readline().strip())
        rest = handle.read().split()
    degn = np.array([int(float(tok)) for tok in rest[:n_blv]], dtype=float)
    raw = np.array([float(tok) for tok in rest[n_blv:]], dtype=float)
    expected = n_blv * nb * nb * 7
    if raw.size != expected:
        raise ValueError(f"Expected {expected} HR values in {path}, found {raw.size}.")
    rows = raw.reshape(n_blv * nb * nb, 7)
    blvd = np.zeros((n_blv, 3), dtype=float)
    hr = np.zeros((n_blv, nb, nb), dtype=complex)
    for i_row, row in enumerate(rows):
        i_r = i_row // (nb * nb)
        iw = int(round(row[3])) - 1
        jw = int(round(row[4])) - 1
        if i_row % (nb * nb) == 0:
            blvd[i_r, :] = row[:3]
        hr[i_r, iw, jw] = row[5] + 1j * row[6]
    return nb, blvd, degn, hr


def _read_wf_centers_spinor(path: str | Path) -> WFMetadata:
    """
    _read_wf_centers_spinor
    
    PURPOSE:
        Read wf centers spinor.
    
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
    raw_lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    header_idx = None
    for i, line in enumerate(raw_lines):
        toks = line.split()
        if len(toks) >= 2 and toks[0].lower() == "i" and toks[1].lower() == "i":
            header_idx = i
            break
    if header_idx is None or header_idx == len(raw_lines) - 1:
        raise ValueError(f"Missing WF_centers table header or data rows in {path}.")
    header = raw_lines[header_idx].split()
    has_spinor = any(tok.lower() == "spinor" for tok in header)
    rows = raw_lines[header_idx + 1 :]
    index = np.zeros(len(rows), dtype=int)
    atom = np.zeros(len(rows), dtype=int)
    spinor = np.empty(len(rows), dtype=object)
    for r, line in enumerate(rows):
        toks = line.split()
        if len(toks) < 2:
            raise ValueError(f"WF_centers row has fewer than two columns: {line}")
        index[r] = int(float(toks[0]))
        atom[r] = int(float(toks[1]))
        spinor[r] = _normalize_spinor_label(toks[2]) if has_spinor and len(toks) >= 3 else ("up" if index[r] % 2 else "down")
    if not np.array_equal(index, np.arange(1, len(rows) + 1)):
        raise ValueError(f"WF indices in {path} must be contiguous 1:NB.")
    return WFMetadata(index=index, atom=atom, spinor=spinor)


def _normalize_spinor_label(label: str) -> str:
    """
    _normalize_spinor_label
    
    PURPOSE:
        Normalize spinor label.
    
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
    s = str(label).strip().lower()
    if s in {"up", "u", "+", "+1", "1"}:
        return "up"
    if s in {"down", "dn", "d", "-", "-1", "2"}:
        return "down"
    raise ValueError(f"Invalid Spinor label {label!r}.")


def _spin_block_reorder_map(spinor: np.ndarray) -> np.ndarray:
    """
    _spin_block_reorder_map
    
    PURPOSE:
        Spin block reorder map.
    
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
    up = [i for i, label in enumerate(spinor) if label == "up"]
    down = [i for i, label in enumerate(spinor) if label == "down"]
    order_old = up + down
    reorder_map = np.zeros(len(spinor), dtype=int)
    for new, old in enumerate(order_old):
        reorder_map[old] = new
    return reorder_map




def _validate_inputs(k: np.ndarray, nb: int, opts: TBOptions, reorder_map: np.ndarray) -> None:
    """
    _validate_inputs
    
    PURPOSE:
        Validate inputs.
    
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
    if k.size == 0:
        raise ValueError("The WBSE kmesh has no k-points.")
    if reorder_map.size != nb or not np.array_equal(np.sort(reorder_map), np.arange(nb)):
        raise ValueError("WF_centers reorder map is not a permutation of 0:NB-1.")
    if opts.vbm < 0 or opts.vbm >= nb - 1:
        raise ValueError("VBM must be between 1 and NB - 1.")
    if opts.cbm <= opts.vbm or opts.cbm >= nb:
        raise ValueError("CBM must be between VBM + 1 and NB.")
    if opts.output_nv < 1 or opts.output_nv > opts.vbm + 1:
        raise ValueError("Requested Nv exceeds the internally selected valence family.")
    if opts.output_nc < 1 or opts.output_nc > nb - opts.cbm:
        raise ValueError("Requested Nc exceeds the internally selected conduction family.")


def _diagonalize_all_kpoints(
    k: np.ndarray,
    blvc: np.ndarray,
    degn: np.ndarray,
    hr: np.ndarray,
    nb: int,
    npar_tb: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    _diagonalize_all_kpoints
    
    PURPOSE:
        Diagonalize all kpoints.
    
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
    nk = k.shape[0]
    e_data = np.zeros((nb, nk), dtype=float)
    c_data = np.zeros((nb, nb, nk), dtype=complex)
    n_workers = max(1, int(npar_tb))
    diagonalization_start = time.perf_counter()
    if n_workers > 1:
        results = Parallel(n_jobs=n_workers, prefer="threads")(
            delayed(_diagonalize_one)(k[ik, :], blvc, degn, hr) for ik in range(nk)
        )
        build_work_s = sum(result[2] for result in results)
        eig_work_s = sum(result[3] for result in results)
        for ik, (energy, vectors, _, _) in enumerate(results):
            e_data[:, ik] = energy
            c_data[:, :, ik] = vectors
    else:
        for ik in range(nk):
            energy, vectors, build_time, eig_time = _diagonalize_one(k[ik, :], blvc, degn, hr)
            e_data[:, ik] = energy
            c_data[:, :, ik] = vectors
            if ik == 0:
                build_work_s = build_time
                eig_work_s = eig_time
            else:
                build_work_s += build_time
                eig_work_s += eig_time
    return e_data, c_data, {
        "wall_s": time.perf_counter() - diagonalization_start,
        "build_work_s": build_work_s,
        "eig_work_s": eig_work_s,
    }


def _diagonalize_one(
    kpt: np.ndarray,
    blvc: np.ndarray,
    degn: np.ndarray,
    hr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    _diagonalize_one
    
    PURPOSE:
        Diagonalize one.
    
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
    build_start = time.perf_counter()
    hk = _build_hk(kpt, blvc, degn, hr)
    build_time = time.perf_counter() - build_start
    eig_start = time.perf_counter()
    eigvals, eigvecs = np.linalg.eig(hk)
    eig_time = time.perf_counter() - eig_start
    order = np.argsort(np.real(eigvals), kind="stable")
    return np.real(eigvals[order]), eigvecs[:, order], build_time, eig_time


def _build_hk(kpt: np.ndarray, blvc: np.ndarray, degn: np.ndarray, hr: np.ndarray) -> np.ndarray:
    """
    _build_hk
    
    PURPOSE:
        Build hk.
    
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
    phase = np.exp(1j * (blvc @ kpt.reshape(3))) / degn
    return np.einsum("r,rij->ij", phase, hr)


def _spin_blocks_after_reorder(
    spinor: np.ndarray,
    reorder_map: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    _spin_blocks_after_reorder
    
    PURPOSE:
        Spin blocks after reorder.
    
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
    reordered = np.empty(len(spinor), dtype=object)
    for old, new in enumerate(reorder_map):
        reordered[new] = spinor[old]
    up = np.array([i for i, label in enumerate(reordered) if label == "up"], dtype=int)
    down = np.array([i for i, label in enumerate(reordered) if label == "down"], dtype=int)
    if up.size == 0 or down.size == 0:
        raise ValueError("WF_centers.txt must contain both up and down spinor labels.")
    return {"up": up, "down": down}


def _spin_expectation(coeff: np.ndarray, spin_blocks: dict[str, np.ndarray]) -> float:
    """
    _spin_expectation
    
    PURPOSE:
        Spin expectation.
    
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
    weight = np.abs(coeff) ** 2
    total = float(np.sum(weight))
    if total <= 0:
        return float("nan")
    return float(np.sum(weight[spin_blocks["up"]]) / total - np.sum(weight[spin_blocks["down"]]) / total)


def _local_band_gaps(e: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    _local_band_gaps
    
    PURPOSE:
        Local band gaps.
    
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
    gaps = np.zeros(len(idx), dtype=float)
    for i, band in enumerate(idx):
        left = np.inf if band <= 0 else abs(e[band] - e[band - 1])
        right = np.inf if band >= len(e) - 1 else abs(e[band + 1] - e[band])
        gaps[i] = min(left, right)
    return gaps


def _spin_adapt_exact_degenerate_clusters(
    e_data: np.ndarray,
    c_data: np.ndarray,
    spin_blocks: dict[str, np.ndarray],
    opts: TBOptions,
) -> np.ndarray:
    """
    _spin_adapt_exact_degenerate_clusters
    
    PURPOSE:
        Spin adapt exact degenerate clusters.
    
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
    nb, nk = e_data.shape
    sz = np.zeros(nb, dtype=float)
    sz[spin_blocks["up"]] = 1.0
    sz[spin_blocks["down"]] = -1.0
    for ik in range(nk):
        first = 0
        while first < nb:
            last = first
            while last + 1 < nb and abs(e_data[last + 1, ik] - e_data[last, ik]) <= opts.exact_rotation_gap_ev:
                last += 1
            if last > first:
                bands = np.arange(first, last + 1)
                u = c_data[:, bands, ik]
                q, _ = np.linalg.qr(u)
                projected = (q.conj().T @ (sz[:, None] * q) + (sz[:, None] * q).conj().T @ q) / 2.0
                values, rotation = np.linalg.eig(projected)
                order = np.argsort(np.real(values))[::-1]
                c_data[:, bands, ik] = q @ rotation[:, order]
            first = last + 1
    return c_data


def _select_output_window_by_band_index(
    e_data: np.ndarray,
    c_data: np.ndarray,
    opts: TBOptions,
    spin_blocks: dict[str, np.ndarray],
) -> TBSelection:
    """
    _select_output_window_by_band_index
    
    PURPOSE:
        Select output window by band index.
    
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
    nb, nk = e_data.shape
    v_template = np.arange(opts.vbm, opts.vbm - opts.output_nv, -1, dtype=int)
    c_template = np.arange(opts.cbm, opts.cbm + opts.output_nc, dtype=int)
    v_idx = np.tile(v_template[:, None], (1, nk))
    c_idx = np.tile(c_template[:, None], (1, nk))
    full_sz = np.zeros((nb, nk), dtype=float)
    v_sz = np.zeros((opts.output_nv, nk), dtype=float)
    c_sz = np.zeros((opts.output_nc, nk), dtype=float)
    v_gap = np.zeros((opts.output_nv, nk), dtype=float)
    c_gap = np.zeros((opts.output_nc, nk), dtype=float)
    for ik in range(nk):
        for ib in range(nb):
            full_sz[ib, ik] = _spin_expectation(c_data[:, ib, ik], spin_blocks)
        v_sz[:, ik] = full_sz[v_idx[:, ik], ik]
        c_sz[:, ik] = full_sz[c_idx[:, ik], ik]
        v_gap[:, ik] = _local_band_gaps(e_data[:, ik], v_idx[:, ik])
        c_gap[:, ik] = _local_band_gaps(e_data[:, ik], c_idx[:, ik])
    return TBSelection(
        v_idx=v_idx, c_idx=c_idx, v_sz=v_sz, c_sz=c_sz,
        v_target_spin=np.sign(v_sz), c_target_spin=np.sign(c_sz),
        spin_paired=np.ones(nk, dtype=bool), spin_ambiguous=np.zeros(nk, dtype=bool),
        overlap_used=np.zeros(nk, dtype=bool), overlap_ref=np.zeros(nk, dtype=float),
        v_overlap=np.full((opts.output_nv, nk), np.nan), c_overlap=np.full((opts.output_nc, nk), np.nan),
        v_overlap_margin=np.full((opts.output_nv, nk), np.nan), c_overlap_margin=np.full((opts.output_nc, nk), np.nan),
        v_gap=v_gap, c_gap=c_gap, full_spectrum_sz=full_sz,
    )


def _enforce_conduction_spin_pairing(
    selection: TBSelection,
    e_data: np.ndarray,
    opts: TBOptions,
) -> TBSelection:
    """
    _enforce_conduction_spin_pairing
    
    PURPOSE:
        Enforce conduction spin pairing.
    
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
    n_pair = min(selection.v_idx.shape[0], selection.c_idx.shape[0])
    if n_pair == 0:
        return selection
    nb, nk = e_data.shape
    pad = max(0, opts.d3h_candidate_padding)
    for ik in range(nk):
        current = selection.c_idx[:, ik].copy()
        candidates = np.arange(opts.cbm, min(nb, int(np.max(current)) + pad + 1), dtype=int)
        target_spin = selection.c_sz[:, ik].copy()
        for i_pair in range(n_pair):
            v_spin = selection.v_sz[i_pair, ik]
            if abs(v_spin) >= opts.spin_threshold:
                target_spin[i_pair] = np.sign(v_spin)
            else:
                selection.spin_ambiguous[ik] = True
        selection.c_target_spin[:, ik] = target_spin
        assignments = _d3h_ordered_assignments(candidates, selection.c_idx.shape[0], selection.full_spectrum_sz[:, ik], target_spin, opts)
        if assignments.size == 0:
            selection.spin_paired[ik] = False
            selection.spin_ambiguous[ik] = True
            continue
        current_energy = e_data[current, ik]
        best_cost = np.inf
        best_assign = current
        for candidate in assignments:
            candidate = candidate.astype(int)
            cost = np.sum(np.abs(e_data[candidate, ik] - current_energy)) / 0.10
            for i_pair in range(n_pair):
                wanted = target_spin[i_pair]
                if wanted != 0:
                    sz = selection.full_spectrum_sz[candidate[i_pair], ik]
                    if abs(sz) >= opts.spin_threshold:
                        cost += 100.0 * abs(np.sign(sz) - wanted)
            if cost < best_cost:
                best_cost = cost
                best_assign = candidate
        selection.c_idx[:, ik] = best_assign
        selection.c_sz[:, ik] = selection.full_spectrum_sz[best_assign, ik]
        selection.c_gap[:, ik] = _local_band_gaps(e_data[:, ik], best_assign)
        for i_pair in range(n_pair):
            v_spin = selection.v_sz[i_pair, ik]
            c_spin = selection.c_sz[i_pair, ik]
            if abs(v_spin) >= opts.spin_threshold and abs(c_spin) >= opts.spin_threshold and np.sign(v_spin) != np.sign(c_spin):
                selection.spin_paired[ik] = False
    return selection


def _d3h_ordered_assignments(
    candidates: np.ndarray,
    n_label: int,
    sz_all: np.ndarray,
    target_spin: np.ndarray,
    opts: TBOptions,
) -> np.ndarray:
    """
    _d3h_ordered_assignments
    
    PURPOSE:
        D3h ordered assignments.
    
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
    pools = []
    for i_label in range(n_label):
        wanted = np.sign(target_spin[i_label])
        pool = np.asarray(candidates, dtype=int)
        if wanted != 0:
            spin_pool = pool[(np.sign(sz_all[pool]) == wanted) & (np.abs(sz_all[pool]) >= opts.spin_threshold)]
            if spin_pool.size == 0:
                spin_pool = pool[np.sign(sz_all[pool]) == wanted]
            if spin_pool.size == 0:
                return np.zeros((0, n_label), dtype=int)
            pool = spin_pool
        pools.append([int(v) for v in pool])
    out = []
    _build_assignments_recursive(pools, n_label, [], set(), out)
    return np.array(out, dtype=int) if out else np.zeros((0, n_label), dtype=int)


def _build_assignments_recursive(
    pools: list[list[int]],
    n_label: int,
    current: list[int],
    used: set[int],
    out: list[tuple[int, ...]],
) -> None:
    """
    _build_assignments_recursive
    
    PURPOSE:
        Build assignments recursive.
    
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
    if len(current) == n_label:
        out.append(tuple(current))
        return
    for candidate in pools[len(current)]:
        if candidate in used:
            continue
        current.append(candidate)
        used.add(candidate)
        _build_assignments_recursive(pools, n_label, current, used, out)
        used.remove(candidate)
        current.pop()



def _enforce_d3h_energy_orbit_consistency(
    selection: TBSelection,
    e_data: np.ndarray,
    k: np.ndarray,
    opts: TBOptions,
) -> TBSelection:
    """
    _enforce_d3h_energy_orbit_consistency
    
    PURPOSE:
        Enforce d3h energy orbit consistency.
    
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
    maps = _find_d3h_partners(k, opts)
    for orbit in _build_symmetry_orbits(maps):
        if len(orbit) < 2:
            continue
        selection = _relabel_d3h_family(selection, e_data, orbit, "v", opts)
        selection = _relabel_d3h_family(selection, e_data, orbit, "c", opts)
    return selection


def _relabel_d3h_family(
    selection: TBSelection,
    e_data: np.ndarray,
    orbit: np.ndarray,
    family: str,
    opts: TBOptions,
) -> TBSelection:
    """
    _relabel_d3h_family
    
    PURPOSE:
        Relabel d3h family.
    
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
    idx = selection.v_idx if family == "v" else selection.c_idx
    sz = selection.v_sz if family == "v" else selection.c_sz
    target = selection.v_target_spin if family == "v" else selection.c_target_spin
    gap = selection.v_gap if family == "v" else selection.c_gap
    n_label = idx.shape[0]
    if n_label == 0 or n_label > 4:
        return selection
    candidates = _d3h_orbit_candidate_window(idx, e_data, orbit, opts)
    reference = int(orbit[0])
    ref_assignments = _d3h_ordered_assignments(candidates, n_label, selection.full_spectrum_sz[:, reference], target[:, reference], opts)
    if ref_assignments.size == 0:
        return selection
    best_cost = np.inf
    best_by_orbit = None
    for ref_assignment in ref_assignments:
        trial = [ref_assignment.astype(int)]
        ref_energy = e_data[trial[0], reference]
        cost = _d3h_assignment_regularization(selection, e_data, reference, family, trial[0], opts)
        for ik in orbit[1:]:
            assignments = _d3h_ordered_assignments(candidates, n_label, selection.full_spectrum_sz[:, ik], target[:, ik], opts)
            if assignments.size == 0:
                cost = np.inf
                break
            local_assign, local_cost = _best_d3h_assignment(selection, e_data, int(ik), family, assignments, ref_energy, opts)
            trial.append(local_assign)
            cost += local_cost
        if np.isfinite(cost) and cost < best_cost:
            best_cost = cost
            best_by_orbit = trial
    if best_by_orbit is None:
        return selection
    current_by_orbit = [idx[:, ik].copy() for ik in orbit]
    current_spread = _d3h_orbit_energy_spread(e_data, orbit, current_by_orbit)
    trial_spread = _d3h_orbit_energy_spread(e_data, orbit, best_by_orbit)
    max_shift = _d3h_max_assignment_shift(e_data, orbit, current_by_orbit, best_by_orbit)
    if trial_spread >= current_spread - opts.d3h_energy_tolerance_ev or max_shift > opts.d3h_max_energy_shift_ev:
        return selection
    for pos, ik in enumerate(orbit):
        idx[:, ik] = best_by_orbit[pos]
        sz[:, ik] = selection.full_spectrum_sz[best_by_orbit[pos], ik]
        gap[:, ik] = _local_band_gaps(e_data[:, ik], best_by_orbit[pos])
    return selection


def _d3h_orbit_candidate_window(
    idx: np.ndarray,
    e_data: np.ndarray,
    orbit: np.ndarray,
    opts: TBOptions,
) -> np.ndarray:
    """
    _d3h_orbit_candidate_window
    
    PURPOSE:
        D3h orbit candidate window.
    
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
    nb = e_data.shape[0]
    pad = max(1, opts.d3h_candidate_padding)
    selected = idx[:, orbit]
    first = max(0, int(np.min(selected)) - pad)
    last = min(nb - 1, int(np.max(selected)) + pad)
    return np.arange(first, last + 1, dtype=int)


def _d3h_candidate_window(e_data: np.ndarray, family: str, opts: TBOptions) -> np.ndarray:
    """
    _d3h_candidate_window
    
    PURPOSE:
        D3h candidate window.
    
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
    nb = e_data.shape[0]
    pad = max(0, opts.d3h_candidate_padding)
    if family == "v":
        return np.arange(0, min(nb - 1, opts.vbm + pad) + 1, dtype=int)
    return np.arange(max(0, opts.cbm - pad), nb, dtype=int)


def _best_d3h_assignment(
    selection: TBSelection,
    e_data: np.ndarray,
    ik: int,
    family: str,
    assignments: np.ndarray,
    reference_energy: np.ndarray,
    opts: TBOptions,
) -> tuple[np.ndarray, float]:
    """
    _best_d3h_assignment
    
    PURPOSE:
        Best d3h assignment.
    
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
    best_cost = np.inf
    best_assign = assignments[0].astype(int)
    for candidate in assignments:
        candidate = candidate.astype(int)
        energy_cost = np.sum(np.abs(e_data[candidate, ik] - reference_energy)) / max(opts.d3h_energy_tolerance_ev, 1.0e-12)
        reg_cost = _d3h_assignment_regularization(selection, e_data, ik, family, candidate, opts)
        cost = energy_cost + reg_cost
        if cost < best_cost:
            best_cost = cost
            best_assign = candidate
    return best_assign, float(best_cost)


def _d3h_assignment_regularization(
    selection: TBSelection,
    e_data: np.ndarray,
    ik: int,
    family: str,
    candidate: np.ndarray,
    opts: TBOptions,
) -> float:
    """
    _d3h_assignment_regularization
    
    PURPOSE:
        D3h assignment regularization.
    
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
    current = selection.v_idx[:, ik] if family == "v" else selection.c_idx[:, ik]
    spin_target = selection.v_target_spin[:, ik] if family == "v" else selection.c_target_spin[:, ik]
    cost = np.sum(np.abs(e_data[candidate, ik] - e_data[current, ik])) / 0.10
    for i_label, band in enumerate(candidate):
        wanted = np.sign(spin_target[i_label])
        if wanted != 0:
            sz = selection.full_spectrum_sz[band, ik]
            if abs(sz) >= opts.spin_threshold:
                cost += 100.0 * abs(np.sign(sz) - wanted)
    return float(cost)


def _d3h_orbit_energy_spread(
    e_data: np.ndarray,
    orbit: np.ndarray,
    assignments: list[np.ndarray],
) -> float:
    """
    _d3h_orbit_energy_spread
    
    PURPOSE:
        D3h orbit energy spread.
    
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
    spread = 0.0
    for label in range(len(assignments[0])):
        energies = np.array([e_data[assignments[pos][label], ik] for pos, ik in enumerate(orbit)])
        spread = max(spread, float(np.max(energies) - np.min(energies)))
    return spread


def _d3h_max_assignment_shift(
    e_data: np.ndarray,
    orbit: np.ndarray,
    current_by_orbit: list[np.ndarray],
    trial_by_orbit: list[np.ndarray],
) -> float:
    """
    _d3h_max_assignment_shift
    
    PURPOSE:
        D3h max assignment shift.
    
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
    max_shift = 0.0
    for pos, ik in enumerate(orbit):
        shift = np.max(np.abs(e_data[current_by_orbit[pos], ik] - e_data[trial_by_orbit[pos], ik]))
        max_shift = max(max_shift, float(shift))
    return max_shift


def _find_d3h_partners(k: np.ndarray, opts: TBOptions) -> np.ndarray:
    """
    _find_d3h_partners
    
    PURPOSE:
        Find d3h partners.
    
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
    nk = k.shape[0]
    maps = np.full((nk, 3), -1, dtype=int)
    reciprocal = np.column_stack([opts.reciprocal[0, :2], opts.reciprocal[1, :2]])
    for op_idx, op in enumerate(_d3h_in_plane_operations()):
        for ik in range(nk):
            target = op @ k[ik, :2]
            best, residual = _nearest_k_mod_reciprocal(k[:, :2], target, reciprocal)
            if residual <= opts.d3h_k_tolerance:
                maps[ik, op_idx] = best
    return maps


def _d3h_in_plane_operations() -> list[np.ndarray]:
    """
    _d3h_in_plane_operations
    
    PURPOSE:
        D3h in plane operations.
    
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
    ops = []
    for n in range(3):
        angle = 2.0 * np.pi * n / 3.0
        ops.append(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
    return ops


def _nearest_k_mod_reciprocal(
    kxy: np.ndarray,
    target: np.ndarray,
    reciprocal: np.ndarray,
) -> tuple[int, float]:
    """
    _nearest_k_mod_reciprocal
    
    PURPOSE:
        Nearest k mod reciprocal.
    
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
    delta = kxy - target
    fractional = np.linalg.solve(reciprocal, delta.T).T
    fractional = fractional - np.round(fractional)
    residual_cart = (reciprocal @ fractional.T).T
    residuals = np.sqrt(np.sum(residual_cart**2, axis=1))
    best = int(np.argmin(residuals))
    return best, float(residuals[best])


def _build_symmetry_orbits(maps: np.ndarray) -> list[np.ndarray]:
    """
    _build_symmetry_orbits
    
    PURPOSE:
        Build symmetry orbits.
    
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
    nk = maps.shape[0]
    visited = np.zeros(nk, dtype=bool)
    orbits = []
    for ik in range(nk):
        if visited[ik]:
            continue
        orbit = set([ik] + [int(v) for v in maps[ik, :] if v >= 0])
        changed = True
        while changed:
            changed = False
            for member in list(orbit):
                linked = [int(v) for v in maps[member, :] if v >= 0]
                old_size = len(orbit)
                orbit.update(linked)
                changed = len(orbit) != old_size
        arr = np.array(sorted(orbit), dtype=int)
        visited[arr] = True
        orbits.append(arr)
    return orbits


def _find_time_reversal_partners(k: np.ndarray, opts: TBOptions) -> np.ndarray:
    """
    _find_time_reversal_partners
    
    PURPOSE:
        Find time reversal partners.
    
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
    nk = k.shape[0]
    partner = np.full(nk, -1, dtype=int)
    b1 = opts.reciprocal[0, :]
    b2 = opts.reciprocal[1, :]
    shifts = [n1 * b1 + n2 * b2 for n1 in range(-1, 2) for n2 in range(-1, 2)]
    for ik in range(nk):
        best = np.inf
        best_index = -1
        base = k[ik, :] + k
        for shift in shifts:
            residuals = np.linalg.norm(base - shift, axis=1)
            local = int(np.argmin(residuals))
            if residuals[local] < best:
                best = float(residuals[local])
                best_index = local
        if best <= opts.tr_k_tolerance:
            partner[ik] = best_index
    return partner



def _enforce_time_reversal_partner_states(
    selection: TBSelection,
    e_data: np.ndarray,
    c_data: np.ndarray,
    k: np.ndarray,
    opts: TBOptions,
) -> TBSelection:
    """
    _enforce_time_reversal_partner_states
    
    PURPOSE:
        Enforce time reversal partner states.
    
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
    partner = _find_time_reversal_partners(k, opts)
    selection.tr_partner = partner
    if selection.tr_relabel_applied is None:
        selection.tr_relabel_applied = np.zeros(k.shape[0], dtype=bool)
    for ik, jk in enumerate(partner):
        if jk < 0 or jk == ik or ik > jk:
            continue
        selection, changed_v = _match_time_reversal_family(selection, e_data, c_data, ik, jk, "v", _d3h_candidate_window(e_data, "v", opts), opts)
        selection, changed_c = _match_time_reversal_family(selection, e_data, c_data, ik, jk, "c", _d3h_candidate_window(e_data, "c", opts), opts)
        selection.tr_relabel_applied[jk] = changed_v or changed_c
    return selection


def _match_time_reversal_family(
    selection: TBSelection,
    e_data: np.ndarray,
    c_data: np.ndarray,
    ik: int,
    jk: int,
    family: str,
    candidates: np.ndarray,
    opts: TBOptions,
) -> tuple[TBSelection, bool]:
    """
    _match_time_reversal_family
    
    PURPOSE:
        Match time reversal family.
    
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
    idx = selection.v_idx if family == "v" else selection.c_idx
    sz = selection.v_sz if family == "v" else selection.c_sz
    gap = selection.v_gap if family == "v" else selection.c_gap
    target_spin_field = selection.v_target_spin if family == "v" else selection.c_target_spin
    source_idx = idx[:, ik]
    original_idx = idx[:, jk].copy()
    target_idx = np.zeros_like(source_idx)
    used = np.zeros(e_data.shape[0], dtype=bool)
    energy_scale = max(opts.tr_energy_tolerance_ev, 1.0e-12)
    for ib, src in enumerate(source_idx):
        available = candidates[~used[candidates]]
        source_energy = e_data[src, ik]
        source_spin = sz[ib, ik]
        target_spin = selection.full_spectrum_sz[available, jk]
        energy_error = np.abs(source_energy - e_data[available, jk])
        if abs(source_spin) >= opts.spin_threshold:
            reliable = np.abs(target_spin) >= opts.spin_threshold
            opposite = reliable & (np.sign(target_spin) == -np.sign(source_spin))
            window = max(opts.tr_energy_tolerance_ev, opts.overlap_degeneracy_gap_ev)
            preferred = opposite & (energy_error <= window)
            if np.any(preferred):
                available = available[preferred]
                target_spin = target_spin[preferred]
                energy_error = energy_error[preferred]
        cost = energy_error / energy_scale
        if abs(source_spin) >= opts.spin_threshold:
            reliable = np.abs(target_spin) >= opts.spin_threshold
            cost[reliable] += np.abs(source_spin + target_spin[reliable])
        target_states = c_data[:, available, jk]
        theta_source = _theta_spin_block(c_data[:, src, ik])
        overlap_cost = 1.0 - np.abs(theta_source.conj().T @ target_states).reshape(-1) ** 2
        cost = cost + 1.0e-3 * overlap_cost
        local = int(np.argmin(cost))
        target_idx[ib] = available[local]
        used[target_idx[ib]] = True
    changed = bool(np.any(target_idx != original_idx))
    idx[:, jk] = target_idx
    sz[:, jk] = selection.full_spectrum_sz[target_idx, jk]
    gap[:, jk] = _local_band_gaps(e_data[:, jk], target_idx)
    target_spin_field[:, jk] = -np.sign(sz[:, ik])
    return selection, changed


def _relabel_time_reversal_pairs(
    selection: TBSelection,
    e_data: np.ndarray,
    k: np.ndarray,
    opts: TBOptions,
) -> TBSelection:
    """
    _relabel_time_reversal_pairs
    
    PURPOSE:
        Relabel time reversal pairs.
    
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
    partner = _find_time_reversal_partners(k, opts)
    selection.tr_partner = partner
    selection.tr_relabel_applied = np.zeros(k.shape[0], dtype=bool)
    for ik, jk in enumerate(partner):
        if jk < 0 or jk == ik or ik > jk:
            continue
        selection, changed_v = _relabel_selected_family(selection, e_data, ik, jk, "v", opts)
        selection, changed_c = _relabel_selected_family(selection, e_data, ik, jk, "c", opts)
        selection.tr_relabel_applied[jk] = changed_v or changed_c
    return selection


def _relabel_selected_family(
    selection: TBSelection,
    e_data: np.ndarray,
    ik: int,
    jk: int,
    family: str,
    opts: TBOptions,
) -> tuple[TBSelection, bool]:
    """
    _relabel_selected_family
    
    PURPOSE:
        Relabel selected family.
    
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
    idx = selection.v_idx if family == "v" else selection.c_idx
    sz = selection.v_sz if family == "v" else selection.c_sz
    target = selection.v_target_spin if family == "v" else selection.c_target_spin
    gap = selection.v_gap if family == "v" else selection.c_gap
    overlap = selection.v_overlap if family == "v" else selection.c_overlap
    margin = selection.v_overlap_margin if family == "v" else selection.c_overlap_margin
    source_idx = idx[:, ik]
    target_idx = idx[:, jk]
    rows = np.arange(len(source_idx), dtype=int)
    if len(source_idx) > 7:
        rows = np.arange(opts.output_nv if family == "v" else opts.output_nc, dtype=int)
    permutation = _best_tr_permutation(e_data[source_idx[rows], ik], e_data[target_idx[rows], jk], sz[rows, ik], sz[rows, jk], opts)
    changed = bool(np.any(permutation != np.arange(len(permutation))))
    old_rows = rows.copy()
    idx[rows, jk] = idx[old_rows[permutation], jk]
    sz[rows, jk] = sz[old_rows[permutation], jk]
    target[rows, jk] = target[old_rows[permutation], jk]
    gap[rows, jk] = gap[old_rows[permutation], jk]
    overlap[rows, jk] = overlap[old_rows[permutation], jk]
    margin[rows, jk] = margin[old_rows[permutation], jk]
    return selection, changed


def _best_tr_permutation(
    e_left: np.ndarray,
    e_right: np.ndarray,
    sz_left: np.ndarray,
    sz_right: np.ndarray,
    opts: TBOptions,
) -> np.ndarray:
    """
    _best_tr_permutation
    
    PURPOSE:
        Best tr permutation.
    
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
    n = len(e_left)
    energy_scale = max(opts.tr_energy_tolerance_ev, 1.0e-12)
    if n <= 7:
        best_cost = None
        best_perm = np.arange(n, dtype=int)
        for candidate in permutations(range(n)):
            candidate_arr = np.array(candidate, dtype=int)
            cost = _tr_permutation_cost(e_left, e_right, sz_left, sz_right, candidate_arr, opts, energy_scale)
            if best_cost is None or cost < best_cost - 10.0 * np.finfo(float).eps * max(1.0, abs(best_cost)):
                best_cost = cost
                best_perm = candidate_arr
        return best_perm
    return _greedy_tr_permutation(e_left, e_right, sz_left, sz_right, opts, energy_scale)


def _tr_permutation_cost(
    e_left: np.ndarray,
    e_right: np.ndarray,
    sz_left: np.ndarray,
    sz_right: np.ndarray,
    candidate: np.ndarray,
    opts: TBOptions,
    energy_scale: float,
) -> float:
    """
    _tr_permutation_cost
    
    PURPOSE:
        Tr permutation cost.
    
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
    cost = np.sum(np.abs(e_left - e_right[candidate])) / energy_scale
    for i, j in enumerate(candidate):
        if abs(sz_left[i]) >= opts.spin_threshold and abs(sz_right[j]) >= opts.spin_threshold:
            cost += abs(sz_left[i] + sz_right[j])
    return float(cost)


def _greedy_tr_permutation(
    e_left: np.ndarray,
    e_right: np.ndarray,
    sz_left: np.ndarray,
    sz_right: np.ndarray,
    opts: TBOptions,
    energy_scale: float,
) -> np.ndarray:
    """
    _greedy_tr_permutation
    
    PURPOSE:
        Greedy tr permutation.
    
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
    n = len(e_left)
    permutation = np.zeros(n, dtype=int)
    unused = np.ones(n, dtype=bool)
    for i in range(n):
        available = np.flatnonzero(unused)
        cost = np.abs(e_left[i] - e_right[available]) / energy_scale
        reliable = (abs(sz_left[i]) >= opts.spin_threshold) & (np.abs(sz_right[available]) >= opts.spin_threshold)
        cost[reliable] += np.abs(sz_left[i] + sz_right[available[reliable]])
        local = int(np.argmin(cost))
        permutation[i] = available[local]
        unused[permutation[i]] = False
    return permutation


def _align_time_reversal_gauge(
    c_data: np.ndarray,
    e_data: np.ndarray,
    selection: TBSelection,
    opts: TBOptions,
) -> np.ndarray:
    """
    _align_time_reversal_gauge
    
    PURPOSE:
        Align time reversal gauge.
    
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
    for ik, jk in enumerate(selection.tr_partner):
        if jk < 0 or jk == ik or ik > jk:
            continue
        c_data = _align_selected_states_in_exact_clusters(c_data, e_data, selection.v_idx[:, ik], selection.v_idx[:, jk], ik, jk, opts)
        c_data = _align_selected_states_in_exact_clusters(c_data, e_data, selection.c_idx[:, ik], selection.c_idx[:, jk], ik, jk, opts)
        c_data = _align_family(c_data, e_data, selection.v_idx[:, ik], selection.v_idx[:, jk], ik, jk, opts)
        c_data = _align_family(c_data, e_data, selection.c_idx[:, ik], selection.c_idx[:, jk], ik, jk, opts)
    return c_data


def _align_selected_states_in_exact_clusters(
    c_data: np.ndarray,
    e_data: np.ndarray,
    idx_left: np.ndarray,
    idx_right: np.ndarray,
    ik: int,
    jk: int,
    opts: TBOptions,
) -> np.ndarray:
    """
    _align_selected_states_in_exact_clusters
    
    PURPOSE:
        Align selected states in exact clusters.
    
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
    for ib, left_band in enumerate(idx_left):
        target = _theta_spin_block(c_data[:, left_band, ik])
        target_band = idx_right[ib]
        first, last = _exact_cluster_bounds(e_data[:, jk], target_band, opts.exact_rotation_gap_ev)
        bands = np.arange(first, last + 1, dtype=int)
        if len(bands) == 1:
            continue
        u = c_data[:, bands, jk]
        coefficients = u.conj().T @ target
        projection = u @ coefficients
        if np.linalg.norm(projection - target) > 1.0e-7:
            continue
        coefficients = coefficients / np.linalg.norm(coefficients)
        local_target = int(np.where(bands == target_band)[0][0])
        rotation = np.zeros((len(bands), len(bands)), dtype=complex)
        rotation[:, local_target] = coefficients
        other = [i for i in range(len(bands)) if i != local_target]
        if other:
            rotation[:, other] = _null_space(coefficients.conj().reshape(1, -1))
        c_data[:, bands, jk] = u @ rotation
    return c_data


def _exact_cluster_bounds(e: np.ndarray, band: int, tolerance: float) -> tuple[int, int]:
    """
    _exact_cluster_bounds
    
    PURPOSE:
        Exact cluster bounds.
    
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
    first = int(band)
    last = int(band)
    while first > 0 and abs(e[first] - e[first - 1]) <= tolerance:
        first -= 1
    while last + 1 < len(e) and abs(e[last + 1] - e[last]) <= tolerance:
        last += 1
    return first, last


def _align_family(
    c_data: np.ndarray,
    e_data: np.ndarray,
    idx_left: np.ndarray,
    idx_right: np.ndarray,
    ik: int,
    jk: int,
    opts: TBOptions,
) -> np.ndarray:
    """
    _align_family
    
    PURPOSE:
        Align family.
    
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
    nb = c_data.shape[0]
    if nb % 2 != 0:
        raise ValueError("TR gauge alignment requires equal spin blocks.")
    first = 0
    while first < len(idx_left):
        last = first
        while (
            last + 1 < len(idx_left)
            and abs(e_data[idx_left[last + 1], ik] - e_data[idx_left[last], ik]) <= opts.exact_rotation_gap_ev
            and abs(e_data[idx_right[last + 1], jk] - e_data[idx_right[last], jk]) <= opts.exact_rotation_gap_ev
        ):
            last += 1
        local = np.arange(first, last + 1, dtype=int)
        target = _theta_spin_block(c_data[:, idx_left[local], ik])
        before = c_data[:, idx_right[local], jk]
        if len(local) == 1:
            overlap = (before.conj().T @ target).item()
            rotation = np.array([[overlap / abs(overlap)]]) if abs(overlap) > 0 else np.array([[1.0]])
        else:
            left, _s, right_h = np.linalg.svd(before.conj().T @ target, full_matrices=False)
            rotation = left @ right_h
        c_data[:, idx_right[local], jk] = before @ rotation
        first = last + 1
    return c_data


def _theta_spin_block(input_data: np.ndarray) -> np.ndarray:
    """
    _theta_spin_block
    
    PURPOSE:
        Theta spin block.
    
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
    half = input_data.shape[0] // 2
    if input_data.ndim == 2:
        return np.vstack([np.conj(input_data[half:, :]), -np.conj(input_data[:half, :])])
    return np.concatenate([np.conj(input_data[half:]), -np.conj(input_data[:half])])


def _null_space(row: np.ndarray, rtol: float = 1.0e-12) -> np.ndarray:
    """
    _null_space
    
    PURPOSE:
        Null space.
    
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
    _u, s, vh = np.linalg.svd(row, full_matrices=True)
    rank = int((s > rtol * s[0]).sum()) if s.size else 0
    return vh[rank:].conj().T


def _time_reversal_diagnostics(e_data: np.ndarray, selection: TBSelection, opts: TBOptions) -> dict:
    """
    _time_reversal_diagnostics
    
    PURPOSE:
        Time reversal diagnostics.
    
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
    partner = selection.tr_partner
    nonself = np.flatnonzero((partner >= 0) & (partner != np.arange(len(partner))))
    n_v = min(opts.output_nv, selection.v_idx.shape[0])
    n_c = min(opts.output_nc, selection.c_idx.shape[0])
    v_energy = np.full((n_v, len(partner)), np.nan)
    c_energy = np.full((n_c, len(partner)), np.nan)
    for ik, jk in enumerate(partner):
        if jk < 0:
            continue
        for ib in range(n_v):
            v_energy[ib, ik] = abs(e_data[selection.v_idx[ib, ik], ik] - e_data[selection.v_idx[ib, jk], jk])
        for ib in range(n_c):
            c_energy[ib, ik] = abs(e_data[selection.c_idx[ib, ik], ik] - e_data[selection.c_idx[ib, jk], jk])
    if nonself.size == 0:
        return {"passed": False, "failure_reason": "no_nonself_time_reversal_pairs", "max_energy_error": np.inf}
    energy_errors = np.vstack([v_energy[:, nonself], c_energy[:, nonself]])
    max_energy = float(np.nanmax(energy_errors))
    reason = "" if max_energy <= opts.tr_energy_tolerance_ev else "time_reversal_energy"
    return {"passed": max_energy <= opts.tr_energy_tolerance_ev, "failure_reason": reason, "max_energy_error": max_energy}


def _refresh_selected_spins(
    c_data: np.ndarray,
    selection: TBSelection,
    spin_blocks: dict[str, np.ndarray],
) -> TBSelection:
    """
    _refresh_selected_spins
    
    PURPOSE:
        Refresh selected spins.
    
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
    for ik in range(selection.v_idx.shape[1]):
        for band in range(c_data.shape[1]):
            selection.full_spectrum_sz[band, ik] = _spin_expectation(c_data[:, band, ik], spin_blocks)
        for ib in range(selection.v_idx.shape[0]):
            selection.v_sz[ib, ik] = _spin_expectation(c_data[:, selection.v_idx[ib, ik], ik], spin_blocks)
        for ib in range(selection.c_idx.shape[0]):
            selection.c_sz[ib, ik] = _spin_expectation(c_data[:, selection.c_idx[ib, ik], ik], spin_blocks)
    return selection


def _write_selected_band_files(
    output_dir: Path,
    k: np.ndarray,
    e_data: np.ndarray,
    c_data: np.ndarray,
    selection: TBSelection,
    opts: TBOptions,
) -> None:
    """
    _write_selected_band_files
    
    PURPOSE:
        Write selected band files.
    
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
    for stale in list(output_dir.glob("v*_TB_WBSE.txt")) + list(output_dir.glob("c*_TB_WBSE.txt")):
        stale.unlink()
    for n in range(opts.output_nv):
        _write_one_selected_band_file(output_dir / f"v{n + 1}_TB_WBSE.txt", k, e_data, c_data, selection.v_idx[n, :])
    for n in range(opts.output_nc):
        _write_one_selected_band_file(output_dir / f"c{n + 1}_TB_WBSE.txt", k, e_data, c_data, selection.c_idx[n, :])


def _write_one_selected_band_file(
    path: Path,
    k: np.ndarray,
    e_data: np.ndarray,
    c_data: np.ndarray,
    selected_bands: np.ndarray,
) -> None:
    """
    _write_one_selected_band_file
    
    PURPOSE:
        Write one selected band file.
    
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
    with path.open("w") as handle:
        for ik, band in enumerate(selected_bands):
            coeff = unify_global_phase_py(c_data[:, band, ik])
            handle.write(f"{k[ik, 0]:20.16f} {k[ik, 1]:20.16f} {k[ik, 2]:20.16f}\n")
            handle.write(f"{e_data[band, ik]:20.16f}\n")
            for value in coeff:
                handle.write(f"{value.real:20.16f} {value.imag:20.16f} {abs(value):20.16f}\n")


def _tb_display_path(path_value: str | Path) -> str:
    """
    _tb_display_path
    
    PURPOSE:
        Tb display path.
    
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
    normalized = str(path_value).replace("\\", "/")
    for anchor in ("Precomputed_data", "User_input", "Parameters", "Exciton_data"):
        pos = normalized.find(anchor)
        if pos >= 0:
            return normalized[pos:]
    return normalized

# The following helpers are the validated Stage 2B-2E2 implementation folded
# into production. Diagnostic tracing and benchmark-only counters are omitted.
def _assignment_cache_key(
    candidates: np.ndarray,
    n_label: int,
    sz_all: np.ndarray,
    target_spin: np.ndarray,
    opts: TBOptions,
) -> tuple:
    """
    _assignment_cache_key
    
    PURPOSE:
        Assignment cache key.
    
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
    return (
        tuple(np.asarray(candidates, dtype=int).tolist()),
        int(n_label),
        tuple(np.asarray(sz_all[np.asarray(candidates, dtype=int)], dtype=float).tolist()),
        tuple(np.asarray(target_spin, dtype=float).tolist()),
        float(opts.spin_threshold),
    )


def _d3h_ordered_assignments_stage2e2(
    candidates: np.ndarray, n_label: int, sz_all: np.ndarray, target_spin: np.ndarray,
    opts: TBOptions, cache: dict[tuple, np.ndarray]
) -> np.ndarray:
    """
    _d3h_ordered_assignments_stage2e2
    
    PURPOSE:
        D3h ordered assignments stage2e2.
    
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
    key = _assignment_cache_key(candidates, n_label, sz_all, target_spin, opts)
    if key in cache:
        return cache[key]
    pools = []
    for i_label in range(n_label):
        wanted = np.sign(target_spin[i_label])
        pool = np.asarray(candidates, dtype=int)
        if wanted != 0:
            spin_pool = pool[(np.sign(sz_all[pool]) == wanted) & (np.abs(sz_all[pool]) >= opts.spin_threshold)]
            if spin_pool.size == 0:
                spin_pool = pool[np.sign(sz_all[pool]) == wanted]
            if spin_pool.size == 0:
                assignments = np.zeros((0, n_label), dtype=int)
                cache[key] = assignments
                return assignments
            pool = spin_pool
        pools.append([int(v) for v in pool])
    out: list[tuple[int, ...]] = []
    _build_assignments_recursive(pools, n_label, [], set(), out)
    assignments = np.array(out, dtype=int) if out else np.zeros((0, n_label), dtype=int)
    cache[key] = assignments
    return assignments


def _prepare_assignment_cost_data(
    selection: TBSelection,
    e_data: np.ndarray,
    ik: int,
    family: str,
    assignments: np.ndarray,
    opts: TBOptions,
) -> dict:
    """
    _prepare_assignment_cost_data
    
    PURPOSE:
        Prepare assignment cost data.
    
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
    current = selection.v_idx[:, ik] if family == "v" else selection.c_idx[:, ik]
    spin_target = selection.v_target_spin[:, ik] if family == "v" else selection.c_target_spin[:, ik]
    candidate_energy = e_data[assignments, ik]
    current_energy = e_data[current, ik]
    regularization = np.sum(np.abs(candidate_energy - current_energy), axis=1) / 0.10
    spin_penalties = []
    for i_label in range(assignments.shape[1]):
        wanted = np.sign(spin_target[i_label])
        if wanted != 0:
            candidate_spin = selection.full_spectrum_sz[assignments[:, i_label], ik]
            reliable = np.abs(candidate_spin) >= opts.spin_threshold
            spin_penalties.append(np.where(
                reliable, 100.0 * np.abs(np.sign(candidate_spin) - wanted), 0.0
            ))
    return {"candidate_energy": candidate_energy, "regularization": regularization, "spin_penalties": spin_penalties}


def _best_d3h_assignment_stage2e2(
    selection: TBSelection, e_data: np.ndarray, ik: int, family: str, assignments: np.ndarray,
    reference_energy: np.ndarray, opts: TBOptions, prepared_cache: dict
) -> tuple[np.ndarray, float]:
    """
    _best_d3h_assignment_stage2e2
    
    PURPOSE:
        Best d3h assignment stage2e2.
    
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
    key = (family, int(ik), id(assignments))
    prepared = prepared_cache.get(key)
    if prepared is None:
        prepared = _prepare_assignment_cost_data(selection, e_data, ik, family, assignments, opts)
        prepared_cache[key] = prepared
    candidate_energy = prepared["candidate_energy"]
    energy_cost = np.sum(np.abs(candidate_energy - reference_energy), axis=1)
    energy_cost = energy_cost / max(opts.d3h_energy_tolerance_ev, 1.0e-12)
    reg_cost = prepared["regularization"].copy()
    # Preserve the validated label-by-label addition order exactly.
    for penalty in prepared["spin_penalties"]:
        reg_cost = reg_cost + penalty
    costs = energy_cost + reg_cost
    winner_position = int(np.argmin(costs))
    return assignments[winner_position].astype(int), float(costs[winner_position])


def _relabel_d3h_family_stage2e2(
    selection: TBSelection, e_data: np.ndarray, orbit: np.ndarray, family: str,
    opts: TBOptions, cache: dict[tuple, np.ndarray]
) -> TBSelection:
    """
    _relabel_d3h_family_stage2e2
    
    PURPOSE:
        Relabel d3h family stage2e2.
    
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
    idx = selection.v_idx if family == "v" else selection.c_idx
    sz = selection.v_sz if family == "v" else selection.c_sz
    target = selection.v_target_spin if family == "v" else selection.c_target_spin
    gap = selection.v_gap if family == "v" else selection.c_gap
    n_label = idx.shape[0]
    if n_label == 0 or n_label > 4:
        return selection
    candidates = _d3h_orbit_candidate_window(idx, e_data, orbit, opts)
    reference = int(orbit[0])
    ref_assignments = _d3h_ordered_assignments_stage2e2(
        candidates, n_label, selection.full_spectrum_sz[:, reference], target[:, reference], opts, cache
    )
    if ref_assignments.size == 0:
        return selection
    best_cost = np.inf
    best_by_orbit = None
    prepared_cache: dict[tuple, dict] = {}
    for ref_assignment in ref_assignments:
        trial = [ref_assignment.astype(int)]
        ref_energy = e_data[trial[0], reference]
        cost = _d3h_assignment_regularization(selection, e_data, reference, family, trial[0], opts)
        for ik in orbit[1:]:
            assignments = _d3h_ordered_assignments_stage2e2(
                candidates, n_label, selection.full_spectrum_sz[:, ik], target[:, ik], opts, cache
            )
            if assignments.size == 0:
                cost = np.inf
                break
            local_assign, local_cost = _best_d3h_assignment_stage2e2(
                selection, e_data, int(ik), family, assignments, ref_energy, opts, prepared_cache
            )
            trial.append(local_assign)
            cost += local_cost
        if np.isfinite(cost) and cost < best_cost:
            best_cost = cost
            best_by_orbit = trial
    if best_by_orbit is None:
        return selection
    current_by_orbit = [idx[:, ik].copy() for ik in orbit]
    current_spread = _d3h_orbit_energy_spread(e_data, orbit, current_by_orbit)
    trial_spread = _d3h_orbit_energy_spread(e_data, orbit, best_by_orbit)
    max_shift = _d3h_max_assignment_shift(e_data, orbit, current_by_orbit, best_by_orbit)
    if trial_spread >= current_spread - opts.d3h_energy_tolerance_ev or max_shift > opts.d3h_max_energy_shift_ev:
        return selection
    for pos, ik in enumerate(orbit):
        idx[:, ik] = best_by_orbit[pos]
        sz[:, ik] = selection.full_spectrum_sz[best_by_orbit[pos], ik]
        gap[:, ik] = _local_band_gaps(e_data[:, ik], best_by_orbit[pos])
    return selection


def _read_mesh_denominator(path: str | Path) -> int:
    """
    _read_mesh_denominator
    
    PURPOSE:
        Read mesh denominator.
    
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
    with Path(path).open() as handle:
        dense_range = int(handle.readline().strip())
        nk_coarse = int(handle.readline().strip())
        nk_dense = int(handle.readline().strip())
    if dense_range < 1 or nk_coarse < 1 or nk_dense < 1:
        raise ValueError("WBSE kmesh header does not contain positive mesh parameters.")
    return nk_coarse * nk_dense


def _fractional_reciprocal_coordinates(k: np.ndarray, reciprocal: np.ndarray) -> np.ndarray:
    """
    _fractional_reciprocal_coordinates
    
    PURPOSE:
        Fractional reciprocal coordinates.
    
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
    matrix = np.column_stack([reciprocal[0, :2], reciprocal[1, :2]])
    return np.linalg.solve(matrix, np.asarray(k[:, :2], dtype=float).T).T


def _canonical_key(frac: np.ndarray, denominator: int) -> tuple[int, int]:
    """
    _canonical_key
    
    PURPOSE:
        Canonical key.
    
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
    scaled = np.rint(np.asarray(frac, dtype=float) * denominator).astype(np.int64)
    return int(scaled[0]), int(scaled[1])


def _build_d3h_index(
    k: np.ndarray,
    opts: TBOptions,
    denominator: int,
) -> dict[tuple[int, int], list[int]]:
    """
    _build_d3h_index
    
    PURPOSE:
        Build d3h index.
    
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
    frac = _fractional_reciprocal_coordinates(k, opts.reciprocal)
    lookup: dict[tuple[int, int], list[int]] = {}
    quantized = np.rint(frac * denominator).astype(np.int64)
    for index, row in enumerate(quantized):
        key = (int(row[0]), int(row[1]))
        lookup.setdefault(key, []).append(index)
    return lookup


def _indexed_partner(
    kxy: np.ndarray, target: np.ndarray, reciprocal: np.ndarray, denominator: int,
    lookup: dict[tuple[int, int], list[int]], source_index: int, rotation: int
) -> tuple[int, float]:
    """
    _indexed_partner
    
    PURPOSE:
        Indexed partner.
    
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
    matrix = np.column_stack([reciprocal[0, :2], reciprocal[1, :2]])
    target_frac = np.linalg.solve(matrix, np.asarray(target, dtype=float))
    key = _canonical_key(target_frac, denominator)
    candidates = lookup.get(key)
    if candidates is None:
        nearest, distance = _nearest_k_mod_reciprocal(kxy, target, matrix)
        raise RuntimeError(
            "D3h indexed lookup key missing: "
            f"source_index={source_index}, rotation={rotation}, target={target.tolist()}, "
            f"key={key}, nearest_baseline_index={nearest}, baseline_distance={distance:.17g}"
        )
    residuals = []
    for candidate in candidates:
        delta = kxy[candidate] - target
        fractional = np.linalg.solve(matrix, delta)
        fractional = fractional - np.round(fractional)
        residuals.append(float(np.linalg.norm(matrix @ fractional)))
    winner = int(np.argmin(np.asarray(residuals)))
    return candidates[winner], residuals[winner]


def _find_d3h_partners_stage2e2(k: np.ndarray, opts: TBOptions, denominator: int) -> np.ndarray:
    """
    _find_d3h_partners_stage2e2
    
    PURPOSE:
        Find d3h partners stage2e2.
    
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
    lookup = _build_d3h_index(k, opts, denominator)
    maps = np.full((k.shape[0], 3), -1, dtype=int)
    kxy = k[:, :2]
    for op_index, op in enumerate(_d3h_in_plane_operations()):
        for ik in range(k.shape[0]):
            candidate, residual = _indexed_partner(kxy, op @ kxy[ik], opts.reciprocal, denominator, lookup, ik, op_index)
            if residual <= opts.d3h_k_tolerance:
                maps[ik, op_index] = candidate
            else:
                raise RuntimeError(
                    f"D3h indexed candidate rejected: source_index={ik}, rotation={op_index}, "
                    f"candidate_index={candidate}, indexed_distance={residual:.17g}, "
                    f"tolerance={opts.d3h_k_tolerance:.17g}"
                )
    return maps


def _enforce_d3h_energy_orbit_consistency_stage2e2(
    selection: TBSelection, e_data: np.ndarray, k: np.ndarray, opts: TBOptions,
    denominator: int, assignment_cache: dict[tuple, np.ndarray]
) -> TBSelection:
    """
    _enforce_d3h_energy_orbit_consistency_stage2e2
    
    PURPOSE:
        Enforce d3h energy orbit consistency stage2e2.
    
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
    maps = _find_d3h_partners_stage2e2(k, opts, denominator)
    for orbit in _build_symmetry_orbits(maps):
        if len(orbit) < 2:
            continue
        selection = _relabel_d3h_family_stage2e2(selection, e_data, orbit, "v", opts, assignment_cache)
        selection = _relabel_d3h_family_stage2e2(selection, e_data, orbit, "c", opts, assignment_cache)
    return selection


def _tr_distance_for_candidate_stage2e2(
    k: np.ndarray,
    source_index: int,
    candidate_index: int,
    reciprocal: np.ndarray,
) -> float:
    """
    _tr_distance_for_candidate_stage2e2
    
    PURPOSE:
        Tr distance for candidate stage2e2.
    
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
    b1 = reciprocal[0, :]
    b2 = reciprocal[1, :]
    base = k[source_index, :] + k[candidate_index, :]
    best = np.inf
    for n1 in range(-1, 2):
        for n2 in range(-1, 2):
            distance = float(np.linalg.norm(base - (n1 * b1 + n2 * b2)))
            if distance < best:
                best = distance
    return best


def _find_time_reversal_partners_stage2e2(
    k: np.ndarray,
    opts: TBOptions,
    denominator: int,
) -> np.ndarray:
    """
    _find_time_reversal_partners_stage2e2
    
    PURPOSE:
        Find time reversal partners stage2e2.
    
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
    lookup = _build_d3h_index(k, opts, denominator)
    partner = np.full(k.shape[0], -1, dtype=int)
    for ik in range(k.shape[0]):
        target = -k[ik, :2]
        frac = _fractional_reciprocal_coordinates(np.asarray([target]), opts.reciprocal)[0]
        key = _canonical_key(frac, denominator)
        candidates = lookup.get(key)
        if candidates is None:
            best = -1
            best_distance = np.inf
            for candidate in range(k.shape[0]):
                distance = _tr_distance_for_candidate_stage2e2(k, ik, candidate, opts.reciprocal)
                if distance < best_distance:
                    best_distance, best = distance, candidate
            raise RuntimeError(
                f"TR indexed lookup key missing: source_index={ik}, source_k={k[ik].tolist()}, "
                f"target={(-k[ik]).tolist()}, canonical_key={key}, nearest_baseline_index={best}, "
                f"baseline_minimum_distance={best_distance:.17g}, tolerance={opts.tr_k_tolerance:.17g}"
            )
        distances = [_tr_distance_for_candidate_stage2e2(k, ik, candidate, opts.reciprocal) for candidate in candidates]
        winner = int(np.argmin(np.asarray(distances)))
        best = candidates[winner]
        best_distance = distances[winner]
        if best_distance <= opts.tr_k_tolerance:
            partner[ik] = best
        else:
            raise RuntimeError(
                f"TR indexed candidate rejected: source_index={ik}, source_k={k[ik].tolist()}, "
                f"target={(-k[ik]).tolist()}, canonical_key={key}, candidate_index={best}, "
                f"baseline_distance={best_distance:.17g}, tolerance={opts.tr_k_tolerance:.17g}"
            )
    return partner


def _enforce_time_reversal_partner_states_stage2a(
    selection: TBSelection, e_data: np.ndarray, c_data: np.ndarray, k: np.ndarray,
    opts: TBOptions, partner: np.ndarray
) -> TBSelection:
    """
    _enforce_time_reversal_partner_states_stage2a
    
    PURPOSE:
        Enforce time reversal partner states stage2a.
    
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
    selection.tr_partner = partner
    if selection.tr_relabel_applied is None:
        selection.tr_relabel_applied = np.zeros(k.shape[0], dtype=bool)
    for ik, jk in enumerate(partner):
        if jk < 0 or jk == ik or ik > jk:
            continue
        selection, changed_v = _match_time_reversal_family(
            selection, e_data, c_data, ik, jk, "v", _d3h_candidate_window(e_data, "v", opts), opts
        )
        selection, changed_c = _match_time_reversal_family(
            selection, e_data, c_data, ik, jk, "c", _d3h_candidate_window(e_data, "c", opts), opts
        )
        selection.tr_relabel_applied[jk] = changed_v or changed_c
    return selection


def _relabel_time_reversal_pairs_stage2a(
    selection: TBSelection, e_data: np.ndarray, opts: TBOptions, partner: np.ndarray
) -> TBSelection:
    """
    _relabel_time_reversal_pairs_stage2a
    
    PURPOSE:
        Relabel time reversal pairs stage2a.
    
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
    selection.tr_partner = partner
    selection.tr_relabel_applied = np.zeros(len(partner), dtype=bool)
    for ik, jk in enumerate(partner):
        if jk < 0 or jk == ik or ik > jk:
            continue
        selection, changed_v = _relabel_selected_family(selection, e_data, ik, jk, "v", opts)
        selection, changed_c = _relabel_selected_family(selection, e_data, ik, jk, "c", opts)
        selection.tr_relabel_applied[jk] = changed_v or changed_c
    return selection
