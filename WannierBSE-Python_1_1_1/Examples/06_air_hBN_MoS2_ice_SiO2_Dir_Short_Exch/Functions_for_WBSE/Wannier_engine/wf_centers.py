"""
wf_centers

PURPOSE:
    Wannier center reordering for the WBSE spin-block convention.

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


def _wf_centers_has_spinor_labels(path: str | Path) -> bool:
    """
    _wf_centers_has_spinor_labels
    
    PURPOSE:
        Return whether a WF-center file has valid up/down spinor labels.
    
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
    if not path.is_file():
        return False
    labels = []
    for line in path.read_text().splitlines():
        stripped = line.split("%", 1)[0].split("#", 1)[0].strip()
        if not stripped:
            continue
        fields = stripped.split()
        try:
            float(fields[0])
            float(fields[1])
        except (ValueError, IndexError):
            continue
        if len(fields) < 3:
            return False
        labels.append(fields[2].strip().lower())
    return bool(labels) and all(label in {"up", "down"} for label in labels)


def read_wf_centers_spinor(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    read_wf_centers_spinor
    
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
    indices = []
    atoms = []
    spinors = []
    for line in Path(path).read_text().splitlines():
        stripped = line.split("%", 1)[0].split("#", 1)[0].strip()
        if not stripped:
            continue
        fields = stripped.split()
        try:
            idx = int(float(fields[0]))
            atom = int(float(fields[1]))
        except (ValueError, IndexError):
            continue
        spinor = fields[2].lower() if len(fields) >= 3 else ""
        indices.append(idx)
        atoms.append(atom)
        spinors.append(spinor)
    if not indices:
        raise ValueError(f"No WF center rows found in {path}")
    return np.array(indices, dtype=int), np.array(atoms, dtype=int), np.array(spinors, dtype=object)


def spin_block_reorder_map(spinors: np.ndarray) -> np.ndarray:
    """
    spin_block_reorder_map
    
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
    labels = np.array([str(label).lower() for label in spinors], dtype=object)
    up = np.flatnonzero(labels == "up")
    down = np.flatnonzero(labels == "down")
    if up.size + down.size != labels.size:
        bad = sorted(set(labels.tolist()) - {"up", "down"})
        raise ValueError(f"Unsupported spinor labels in WF centers: {bad}")
    if up.size != down.size:
        raise ValueError(f"Spinor WF centers must contain equal up/down counts; got up={up.size}, down={down.size}")
    order = np.empty(labels.size, dtype=int)
    new_order_old_indices = np.concatenate([up, down])
    for new_idx, old_idx in enumerate(new_order_old_indices):
        order[old_idx] = new_idx
    return order


def reorder_wf_centers_py(wf_centers_in: str | Path, wf_centers_out: str | Path) -> np.ndarray:
    """
    reorder_wf_centers_py
    
    PURPOSE:
        Reorder wf centers py.
    
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
    _, atoms, spinors = read_wf_centers_spinor(wf_centers_in)
    reorder_map = spin_block_reorder_map(spinors)
    nb = spinors.size
    out_atoms = np.zeros(nb, dtype=int)
    out_spinors = np.empty(nb, dtype=object)
    for old_idx, new_idx in enumerate(reorder_map):
        out_atoms[new_idx] = atoms[old_idx]
        out_spinors[new_idx] = spinors[old_idx]

    wf_centers_out = Path(wf_centers_out)
    wf_centers_out.parent.mkdir(parents=True, exist_ok=True)
    with wf_centers_out.open("w") as handle:
        handle.write("   i     I     Spinor\n")
        for idx, (atom, spinor) in enumerate(zip(out_atoms, out_spinors), start=1):
            handle.write(f"  {idx:3d}   {int(atom):3d}     {spinor}\n")
    return reorder_map
