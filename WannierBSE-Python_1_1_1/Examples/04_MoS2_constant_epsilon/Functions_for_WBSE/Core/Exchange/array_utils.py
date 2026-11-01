"""
array_utils

PURPOSE:
    Array helpers for MATLAB-compatible exchange translation.

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

import h5py
import numpy as np


def unique_tol(values, tol=1.0e-10, by_rows=False):
    """
    unique_tol
    
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
    arr = np.asarray(values)
    if arr.size == 0:
        return arr.copy()
    if by_rows:
        arr2 = np.asarray(arr, dtype=float)
        order = np.lexsort(tuple(arr2[:, col] for col in range(arr2.shape[1] - 1, -1, -1)))
        sorted_arr = arr2[order]
        kept = [sorted_arr[0]]
        indices = [order[0]]
        for idx, row in zip(order[1:], sorted_arr[1:]):
            if np.max(np.abs(row - kept[-1])) > tol:
                kept.append(row)
                indices.append(idx)
        return np.asarray(kept), np.asarray(indices, dtype=int)
    flat = np.sort(np.asarray(arr, dtype=float).reshape(-1))
    kept = [flat[0]]
    for value in flat[1:]:
        if abs(value - kept[-1]) > tol:
            kept.append(value)
    return np.asarray(kept)


def matlab_complex_to_numpy(data):
    """
    matlab_complex_to_numpy
    
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


def matlab_complex_dtype():
    """
    matlab_complex_dtype
    
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


def as_matlab_complex(data):
    """
    as_matlab_complex
    
    PURPOSE:
        As matlab complex.
    
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
    values = np.asarray(data)
    out = np.empty(values.shape, dtype=matlab_complex_dtype())
    out["real"] = values.real
    out["imag"] = values.imag
    return out


def write_matlab_dataset(path, name, data, compression="gzip", compression_opts=4):
    """
    write_matlab_dataset
    
    PURPOSE:
        Write matlab dataset.
    
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
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(data)
    stored = as_matlab_complex(values).transpose(tuple(reversed(range(values.ndim)))) if np.iscomplexobj(values) else values.T
    with h5py.File(path, "w") as h5:
        dset = h5.create_dataset(name, data=stored, compression=compression, compression_opts=compression_opts, shuffle=True)
        dset.attrs["MATLAB_class"] = np.bytes_("double")
