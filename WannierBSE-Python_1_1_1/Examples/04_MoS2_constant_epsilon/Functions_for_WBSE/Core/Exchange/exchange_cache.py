"""
exchange_cache

PURPOSE:
    Load precomputed short-range exchange tensors.

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

import h5py
import numpy as np


def _matlab_complex_to_numpy(data):
    """
    _matlab_complex_to_numpy
    
    PURPOSE:
        Matlab complex to numpy.
    
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
    if getattr(data.dtype, "fields", None) and "real" in data.dtype.fields and "imag" in data.dtype.fields:
        return data["real"] + 1j * data["imag"]
    return np.asarray(data)


def load_precomputed_exchange_py(exchange_path: str | Path, expected_nb: int | None = None):
    """
    load_precomputed_exchange_py
    
    PURPOSE:
        Load MATLAB v7.3 ``chi_WBSE.mat`` and ``R_mNN_WBSE.mat``.
    
    DESCRIPTION:
        HDF5 stores MATLAB arrays with reversed dimensions, so this loader
        transposes datasets back to MATLAB logical shapes before returning them.
    
    INPUT ARGUMENTS:
        See the annotated signature and detailed description.
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """

    exchange_path = Path(exchange_path)
    chi_file = exchange_path / "chi_WBSE.mat"
    r_file = exchange_path / "R_mNN_WBSE.mat"
    if not chi_file.is_file():
        raise FileNotFoundError(f"Precomputed chi file not found: {chi_file}")
    if not r_file.is_file():
        raise FileNotFoundError(f"Precomputed R_mNN file not found: {r_file}")

    with h5py.File(chi_file, "r") as h5:
        if "chi" not in h5:
            raise ValueError(f"File does not contain dataset chi: {chi_file}")
        chi_raw = _matlab_complex_to_numpy(h5["chi"][()])
    chi = np.transpose(chi_raw, tuple(reversed(range(chi_raw.ndim))))

    with h5py.File(r_file, "r") as h5:
        if "R_mNN" not in h5:
            raise ValueError(f"File does not contain dataset R_mNN: {r_file}")
        r_raw = np.asarray(h5["R_mNN"][()], dtype=float)
    R_mNN = r_raw.T if r_raw.shape[0] == 3 else r_raw

    if chi.ndim != 6:
        raise ValueError(f"chi must be rank-6 after MATLAB transpose, got {chi.shape}")
    if R_mNN.ndim != 2 or R_mNN.shape[1] != 3:
        raise ValueError(f"R_mNN must have shape (NR, 3), got {R_mNN.shape}")
    if chi.shape[4] != R_mNN.shape[0] or chi.shape[5] != R_mNN.shape[0]:
        raise ValueError(f"chi/R_mNN incompatible: chi shape {chi.shape}, R_mNN shape {R_mNN.shape}")
    if expected_nb is not None and tuple(chi.shape[:4]) != (expected_nb, expected_nb, expected_nb, expected_nb):
        raise ValueError(f"chi basis dimensions {chi.shape[:4]} do not match expected NB={expected_nb}")

    chi_p = np.transpose(chi, (2, 3, 0, 1, 4, 5))
    return chi, chi_p, R_mNN
