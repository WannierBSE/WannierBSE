"""
find_j_index_in_rho_integral

PURPOSE:
    Basis-neighborhood helper for exchange rho/chi integrals.

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


def find_j_index_in_rho_integral_py(R, tau, n_atoms, d_max, basis_tau_index, mmax_tolerance):
    """
    find_j_index_in_rho_integral_py
    
    PURPOSE:
        Find j index in rho integral py.
    
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
    R = np.asarray(R, dtype=float)
    tau = np.asarray(tau, dtype=float)
    # MATLAB cell arrays auto-expand when indexed by Wannier basis ID.
    # Python lists do not, so allocate the basis-indexed output explicitly.
    nb = max((max(indices) for indices in basis_tau_index if indices), default=0)
    out = [[] for _ in range(nb)]
    for ni in range(int(n_atoms)):
        tau_p_index = [idx - 1 for idx in basis_tau_index[ni]]
        d = np.linalg.norm(R[:2] + tau[:, :2] - tau[ni, :2], axis=1)
        atom_hits = np.flatnonzero(d <= (d_max + mmax_tolerance))
        idxs = []
        for atom in atom_hits:
            idxs.extend([idx - 1 for idx in basis_tau_index[int(atom)]])
        idxs = sorted(set(idxs))
        for ii in tau_p_index:
            out[ii] = idxs
    return out
