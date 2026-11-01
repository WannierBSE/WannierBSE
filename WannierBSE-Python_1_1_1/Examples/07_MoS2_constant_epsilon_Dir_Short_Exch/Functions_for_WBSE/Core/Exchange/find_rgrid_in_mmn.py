"""
find_rgrid_in_mmn

PURPOSE:
    Enumerate m-nearest-neighbor lattice translations for exchange.

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

import numpy as np
from .array_utils import unique_tol


def find_rgrid_in_mmn_py(
    a1,
    a2,
    tau,
    n_atoms,
    supercell_size,
    mmax,
    nn_distance_tolerance,
    mmax_tolerance,
):
    """
    find_rgrid_in_mmn_py
    
    PURPOSE:
        Find rgrid in mmn py.
    
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
    tau = np.asarray(tau, dtype=float)
    sc = int(supercell_size)
    rows = []
    rgrid = []
    center = (sc - 1) / 2 if sc % 2 else sc / 2
    for i in range(sc):
        for j in range(sc):
            n1, n2 = i - center, j - center
            rows.append([n1, n2])
            rgrid.append(n1 * a1 + n2 * a2)
    n1n2_grid = np.asarray(rows, dtype=float)
    R_grid = np.asarray(rgrid, dtype=float)

    atom_position = []
    R_for_atoms = []
    n_for_atoms = []
    for rg, nn in zip(R_grid, n1n2_grid):
        for atom in range(int(n_atoms)):
            atom_position.append(tau[atom] + rg)
            R_for_atoms.append(rg)
            n_for_atoms.append(nn)
    atom_position = np.asarray(atom_position)
    R_for_atoms = np.asarray(R_for_atoms)
    n_for_atoms = np.asarray(n_for_atoms)

    sorted_d = []
    distance_info = []
    for atom in range(int(n_atoms)):
        d = np.linalg.norm(atom_position[:, :2] - tau[atom, :2], axis=1)
        order = np.argsort(d)
        sorted_di = d[order]
        sorted_d.append(sorted_di)
        distance_info.append((sorted_di, R_for_atoms[order], n_for_atoms[order]))

    dm_all = []
    for sorted_di in sorted_d:
        vals = []
        for value in sorted_di:
            same = sorted_di[np.abs(sorted_di - value) <= nn_distance_tolerance]
            if same.size >= 2:
                vals.append(np.max(same))
        if vals:
            dm_all.extend(unique_tol(np.asarray(vals), 1.0e-10).tolist())
    dm_all = np.sort(np.asarray(dm_all, dtype=float))
    vals = []
    for value in dm_all:
        same = dm_all[np.abs(dm_all - value) <= nn_distance_tolerance]
        if same.size >= 2:
            vals.append(np.max(same))
    dm = unique_tol(np.asarray(vals), 1.0e-10)
    d_max = float(dm[int(mmax)])

    n_temp = []
    r_temp = []
    for sorted_di, r_sorted, n_sorted in distance_info:
        hits = np.flatnonzero(sorted_di <= d_max + mmax_tolerance)
        n_temp.extend(n_sorted[hits])
        r_temp.extend(r_sorted[hits])
    n_temp = np.asarray(n_temp, dtype=float)
    r_temp = np.asarray(r_temp, dtype=float)
    n_unique, ia = unique_tol(n_temp, 1.0e-4, by_rows=True)
    r_mnn = r_temp[ia]
    return r_mnn, n_unique, d_max
