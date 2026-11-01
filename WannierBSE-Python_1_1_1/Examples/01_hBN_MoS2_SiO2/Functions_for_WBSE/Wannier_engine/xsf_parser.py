"""
xsf_parser

PURPOSE:
    XSF parsing helpers for Wannier spinor projections.

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

import mmap
import time
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


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
class XSFGrid:
    """
    XSFGrid
    
    PURPOSE:
        Configuration or data container used by xsf_parser.
    
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
    path: Path
    atoms: list[tuple[str, np.ndarray]]
    dims: tuple[int, int, int]
    origin: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    a3: np.ndarray
    data_start: int
    data_end: int

    @property
    def n_points(self) -> int:
        """
        n_points
        
        PURPOSE:
            N points.
        
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
        return int(np.prod(self.dims))


def natural_key(path: Path) -> tuple:
    """
    natural_key
    
    PURPOSE:
        Natural key.
    
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
    parts = []
    token = ""
    for char in path.name:
        if char.isdigit():
            token += char
        else:
            if token:
                parts.append(int(token))
                token = ""
            parts.append(char)
    if token:
        parts.append(int(token))
    return tuple(parts)


def list_xsf_files(folder: str | Path) -> list[Path]:
    # OPT-INSTRUMENTATION
    """
    list_xsf_files
    
    PURPOSE:
        List xsf files.
    
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
    t_opt = time.perf_counter()
    files = sorted(Path(folder).glob("*.xsf"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No .xsf files found in {folder}")
    _opt_emit("xsf_discovery", time.perf_counter() - t_opt, "exclusive", str(len(files)), "NA")
    return files


def parse_xsf_header(path: str | Path) -> XSFGrid:
    # OPT-INSTRUMENTATION
    """
    parse_xsf_header
    
    PURPOSE:
        Parse xsf header.
    
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
    t_opt = time.perf_counter()
    path = Path(path)
    header_lines = []
    end_grid = None
    with path.open(errors="replace") as handle:
        for idx, line in enumerate(handle):
            header_lines.append(line.rstrip("\n\r"))
            if "END_DATAGRID_3D" in line.upper():
                end_grid = idx
                break
    lines = header_lines
    primcoord = next((i for i, line in enumerate(lines) if line.strip().upper() == "PRIMCOORD"), None)
    begin_grid = next((i for i, line in enumerate(lines) if "BEGIN_DATAGRID_3D" in line.upper()), None)
    if primcoord is None or begin_grid is None or end_grid is None:
        raise ValueError(f"Cannot locate PRIMCOORD/DATAGRID block in {path}")

    atom_count = int(lines[primcoord + 1].split()[0])
    atoms = []
    for line in lines[primcoord + 2 : primcoord + 2 + atom_count]:
        fields = line.split()
        atoms.append((fields[0], np.array([float(x) for x in fields[1:4]], dtype=float)))

    dims = tuple(int(float(x)) for x in lines[begin_grid + 1].split()[:3])
    origin = np.fromstring(lines[begin_grid + 2], sep=" ", dtype=float)[:3]
    a1 = np.fromstring(lines[begin_grid + 3], sep=" ", dtype=float)[:3]
    a2 = np.fromstring(lines[begin_grid + 4], sep=" ", dtype=float)[:3]
    a3 = np.fromstring(lines[begin_grid + 5], sep=" ", dtype=float)[:3]
    if any(len(vec) != 3 for vec in (origin, a1, a2, a3)):
        raise ValueError(f"Malformed DATAGRID geometry in {path}")
    grid = XSFGrid(path, atoms, dims, origin, a1, a2, a3, begin_grid + 6, end_grid)
    _opt_emit("xsf_header_parse", time.perf_counter() - t_opt, "exclusive", "1", str(dims))
    return grid


def read_xsf_values_reference(path: str | Path, expected_points: int | None = None) -> np.ndarray:
    """
    read_xsf_values_reference
    
    PURPOSE:
        Reference line-list parser retained for regression comparisons.
    
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
    grid = parse_xsf_header(path)
    expected = grid.n_points if expected_points is None else int(expected_points)
    chunks = []
    with grid.path.open(errors="replace") as handle:
        for idx, line in enumerate(handle):
            if grid.data_start <= idx < grid.data_end:
                chunks.append(line)
            elif idx >= grid.data_end:
                break
    payload = "".join(chunks)
    values = np.fromstring(payload, sep=" ", dtype=np.float64)
    if values.size != expected:
        raise ValueError(f"{grid.path} contains {values.size} grid values; expected {expected}")
    return values


def _numeric_body_bounds(raw: bytes, path: Path) -> tuple[int, int]:
    """
    _numeric_body_bounds
    
    PURPOSE:
        Locate the numeric DATAGRID body without materializing Python lines.
    
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
    begin_start = raw.find(b"BEGIN_DATAGRID_3D")
    begin_end = begin_start + len(b"BEGIN_DATAGRID_3D") if begin_start >= 0 else -1
    if begin_end < 0:
        begin = re.search(rb"BEGIN_DATAGRID_3D", raw, flags=re.IGNORECASE)
        if begin is None:
            raise ValueError(f"Cannot locate BEGIN_DATAGRID_3D in {path}")
        begin_end = begin.end()
    cursor = raw.find(b"\n", begin_end)
    if cursor < 0:
        raise ValueError(f"Malformed DATAGRID header in {path}")
    for _ in range(5):
        cursor = raw.find(b"\n", cursor + 1)
        if cursor < 0:
            raise ValueError(f"Malformed DATAGRID geometry in {path}")
    start = cursor + 1
    end = raw.find(b"END_DATAGRID_3D", start)
    if end < 0:
        end_match = re.search(rb"END_DATAGRID_3D", raw[start:], flags=re.IGNORECASE)
        if end_match is None:
            raise ValueError(f"Cannot locate END_DATAGRID_3D in {path}")
        end = start + end_match.start()
    return start, end


def read_xsf_values_fast(path: str | Path, expected_points: int | None = None) -> np.ndarray:
    """
    read_xsf_values_fast
    
    PURPOSE:
        Read XSF values without a Python line list or join.
    
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
    path = Path(path)
    raw = path.read_bytes()
    start, end = _numeric_body_bounds(raw, path)
    header = raw[:start].decode(errors="replace")
    lines = header.splitlines()
    begin_grid = next((i for i, line in enumerate(lines) if "BEGIN_DATAGRID_3D" in line.upper()), None)
    if begin_grid is None:
        raise ValueError(f"Cannot locate PRIMCOORD/DATAGRID block in {path}")
    primcoord = next((i for i, line in enumerate(lines) if line.strip().upper() == "PRIMCOORD"), None)
    if primcoord is None:
        raise ValueError(f"Cannot locate PRIMCOORD/DATAGRID block in {path}")
    atom_count = int(lines[primcoord + 1].split()[0])
    for line in lines[primcoord + 2 : primcoord + 2 + atom_count]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed PRIMCOORD atom data in {path}")
        [float(x) for x in fields[1:4]]
    dims = tuple(int(float(x)) for x in lines[begin_grid + 1].split()[:3])
    if len(dims) != 3 or any(value <= 0 for value in dims):
        raise ValueError(f"Malformed DATAGRID dimensions in {path}")
    for line_index in range(begin_grid + 2, begin_grid + 6):
        if np.fromstring(lines[line_index], sep=" ", dtype=float).size < 3:
            raise ValueError(f"Malformed DATAGRID geometry in {path}")
    parsed_expected = int(np.prod(dims))
    expected = parsed_expected if expected_points is None else int(expected_points)
    if expected != parsed_expected:
        raise ValueError(f"{path} grid dimensions imply {parsed_expected} values; expected {expected}")
    values = np.fromstring(raw[start:end], sep=" ", dtype=np.float64)
    if values.size != expected:
        raise ValueError(f"{path} contains {values.size} grid values; expected {expected}")
    return values


def read_xsf_values_mmap(path: str | Path, expected_points: int | None = None) -> np.ndarray:
    """
    read_xsf_values_mmap
    
    PURPOSE:
        Read XSF values through a bounded memory map.
    
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
    path = Path(path)
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        start, end = _numeric_body_bounds(raw, path)
        header = bytes(raw[:start]).decode(errors="replace")
        lines = header.splitlines()
        begin_grid = next((i for i, line in enumerate(lines) if "BEGIN_DATAGRID_3D" in line.upper()), None)
        if begin_grid is None:
            raise ValueError(f"Cannot locate PRIMCOORD/DATAGRID block in {path}")
        primcoord = next((i for i, line in enumerate(lines) if line.strip().upper() == "PRIMCOORD"), None)
        if primcoord is None:
            raise ValueError(f"Cannot locate PRIMCOORD/DATAGRID block in {path}")
        atom_count = int(lines[primcoord + 1].split()[0])
        for line in lines[primcoord + 2 : primcoord + 2 + atom_count]:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed PRIMCOORD atom data in {path}")
            [float(x) for x in fields[1:4]]
        dims = tuple(int(float(x)) for x in lines[begin_grid + 1].split()[:3])
        if len(dims) != 3 or any(value <= 0 for value in dims):
            raise ValueError(f"Malformed DATAGRID dimensions in {path}")
        for line_index in range(begin_grid + 2, begin_grid + 6):
            if np.fromstring(lines[line_index], sep=" ", dtype=float).size < 3:
                raise ValueError(f"Malformed DATAGRID geometry in {path}")
        parsed_expected = int(np.prod(dims))
        expected = parsed_expected if expected_points is None else int(expected_points)
        if expected != parsed_expected:
            raise ValueError(f"{path} grid dimensions imply {parsed_expected} values; expected {expected}")
        values = np.fromstring(raw[start:end], sep=" ", dtype=np.float64)
        if values.size != expected:
            raise ValueError(f"{path} contains {values.size} grid values; expected {expected}")
        return values


def read_xsf_values(path: str | Path, expected_points: int | None = None) -> np.ndarray:
    """
    read_xsf_values
    
    PURPOSE:
        Read XSF values with the default memory-mapped parser.
    
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
    # OPT-INSTRUMENTATION
    t_opt = time.perf_counter()
    values = read_xsf_values_mmap(path, expected_points=expected_points)
    _opt_emit("xsf_values_read", time.perf_counter() - t_opt, "exclusive", "1", str(values.shape))
    return values


def make_cartesian_grid(grid: XSFGrid) -> np.ndarray:
    # OPT-INSTRUMENTATION
    """
    make_cartesian_grid
    
    PURPOSE:
        Make cartesian grid.
    
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
    t_opt = time.perf_counter()
    nx, ny, nz = grid.dims
    ix = np.tile(np.arange(nx, dtype=float), ny * nz)
    iy = np.tile(np.repeat(np.arange(ny, dtype=float), nx), nz)
    iz = np.repeat(np.arange(nz, dtype=float), nx * ny)
    r = grid.origin.reshape(1, 3)
    r = r + (ix / (nx - 1)).reshape(-1, 1) * grid.a1.reshape(1, 3)
    r = r + (iy / (ny - 1)).reshape(-1, 1) * grid.a2.reshape(1, 3)
    r = r + (iz / (nz - 1)).reshape(-1, 1) * grid.a3.reshape(1, 3)
    _opt_emit("cartesian_grid_build", time.perf_counter() - t_opt, "exclusive", "1", str(r.shape))
    return r
