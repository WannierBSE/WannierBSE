"""
exchange_coulomb_interaction

PURPOSE:
    Short-range exchange Coulomb contraction for WannierBSE.

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

import os
import numpy as np
import time


_PACKED_CHI_MEMMAP_CACHE = {}


# ==========================================================================
# PREPARE_PACKED_CHI_MEMMAP
#
# DESCRIPTION:
#   Converts a precomputed exchange tensor chi_p into a single C-order,
#   file-backed matrix collection for reuse by exchange workers.
#
# INPUT ARGUMENTS:
#   chi_p        : complex exchange tensor, shape (NB, NB, NB, NB, nr, nr).
#   backing_path : unique path for the read-only worker backing file.
#
# OUTPUT:
#   descriptor : path, shape, dtype, byte count, and packing duration.
#
# METHODOLOGY:
#   Each (r1, r2) slice is reshaped as (NB, NB**3), preserving the S1
#   index order.  Packing occurs once per calculation; workers receive only
#   this descriptor and never serialize or reconstruct the full packed array.
#
# NOTES:
#   chi_p is assumed already calculated.  The caller owns backing-file
#   cleanup after all workers have completed.  complex128 is required.
# ==========================================================================
def prepare_packed_chi_memmap(chi_p, backing_path):
    """
    prepare_packed_chi_memmap
    
    PURPOSE:
        Create a fresh C-order packed chi backing file in the parent process.
    
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
    chi_p = np.asarray(chi_p)
    if chi_p.ndim != 6 or len(set(chi_p.shape[:4])) != 1:
        raise ValueError(f"chi_p must have four equal basis axes, got {chi_p.shape}")
    if chi_p.shape[4] != chi_p.shape[5]:
        raise ValueError(f"chi_p real-space axes must be equal, got {chi_p.shape}")

    nb = chi_p.shape[0]
    nr = chi_p.shape[4]
    shape = (nr * nr, nb, nb ** 3)
    backing_path = os.fspath(backing_path)
    if os.path.exists(backing_path):
        raise FileExistsError(f"Refusing to reuse stale packed chi file: {backing_path}")

    started = time.perf_counter()
    packed = None
    try:
        packed = np.memmap(backing_path, dtype=np.complex128, mode="w+", shape=shape, order="C")
        for r1 in range(nr):
            for r2 in range(nr):
                packed[r1 * nr + r2] = chi_p[:, :, :, :, r1, r2].reshape(nb, nb ** 3)
        packed.flush()
        packed.flags.writeable = False
        elapsed = time.perf_counter() - started
        expected_bytes = int(np.prod(shape, dtype=np.int64) * np.dtype(np.complex128).itemsize)
        actual_bytes = os.path.getsize(backing_path)
        if actual_bytes != expected_bytes:
            raise ValueError(f"Packed chi size {actual_bytes} != expected {expected_bytes}")
    except Exception:
        if packed is not None:
            del packed
        try:
            os.unlink(backing_path)
        except OSError:
            pass
        raise
    del packed
    return {
        "path": os.path.abspath(backing_path),
        "shape": tuple(int(dim) for dim in shape),
        "dtype": np.dtype(np.complex128).str,
        "bytes": actual_bytes,
        "packing_s": elapsed,
    }


# ==========================================================================
# _GET_PACKED_CHI_MEMMAP
#
# DESCRIPTION:
#   Opens and caches one read-only packed chi map in a loky worker.
#
# INPUT ARGUMENTS:
#   descriptor : parent-created packed-chi descriptor.
#   nb, nr     : runtime Wannier and real-space dimensions.
#
# OUTPUT:
#   packed, created, open_seconds : cached read-only map and access metadata.
#
# METHODOLOGY:
#   The descriptor is validated against runtime dimensions and the backing
#   file, then used as a per-process cache key across source-group batches.
#
# NOTES:
#   This function does not repack chi and never permits worker writes.
# ==========================================================================
def _get_packed_chi_memmap(descriptor, nb, nr):
    """
    _get_packed_chi_memmap
    
    PURPOSE:
        Open a descriptor's read-only map once per persistent worker process.
    
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
    if descriptor is None:
        return None, False, 0.0
    path = os.path.abspath(os.fspath(descriptor["path"]))
    shape = tuple(int(dim) for dim in descriptor["shape"])
    dtype = np.dtype(descriptor["dtype"])
    expected_shape = (nr * nr, nb, nb ** 3)
    if shape != expected_shape or dtype != np.dtype(np.complex128):
        raise ValueError(
            f"Packed chi descriptor {(shape, dtype)} does not match "
            f"{(expected_shape, np.dtype(np.complex128))}"
        )
    expected_bytes = int(np.prod(shape, dtype=np.int64) * dtype.itemsize)
    if os.path.getsize(path) != expected_bytes:
        raise ValueError(f"Packed chi backing file has invalid size: {path}")
    key = (path, shape, dtype.str)
    packed = _PACKED_CHI_MEMMAP_CACHE.get(key)
    if packed is not None:
        return packed, False, 0.0
    started = time.perf_counter()
    packed = np.memmap(path, dtype=dtype, mode="r", shape=shape, order="C")
    _PACKED_CHI_MEMMAP_CACHE[key] = packed
    return packed, True, time.perf_counter() - started


def exchange_coulomb_interaction_py(chi_p, R_mNN, k, N_kpt, Cv, Cc, c, cc, v, vv, j):
    """
    exchange_coulomb_interaction_py
    
    PURPOSE:
        Contract the precomputed exchange tensor into one Vx k-column.
    
    DESCRIPTION:
        This mirrors MATLAB ``exchange_coulomb_interaction.m``. Band indices
        ``c, cc, v, vv`` are 1-based to match the surrounding WBSE code.
    
    INPUT ARGUMENTS:
        See the annotated signature and detailed description.
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """

    chi_p = np.asarray(chi_p)
    R_mNN = np.asarray(R_mNN, dtype=float)
    k = np.asarray(k, dtype=float)
    Cv = np.asarray(Cv)
    Cc = np.asarray(Cc)

    if chi_p.ndim != 6:
        raise ValueError(f"chi_p must be rank-6, got shape {chi_p.shape}")
    if R_mNN.ndim != 2 or R_mNN.shape[1] < 2:
        raise ValueError(f"R_mNN must have shape (NR, >=2), got {R_mNN.shape}")
    if chi_p.shape[4] != R_mNN.shape[0] or chi_p.shape[5] != R_mNN.shape[0]:
        raise ValueError(f"chi_p/R_mNN mismatch: chi_p shape {chi_p.shape}, R_mNN shape {R_mNN.shape}")

    c0, cc0, v0, vv0 = c - 1, cc - 1, v - 1, vv - 1
    nr = R_mNN.shape[0]
    vx = np.zeros(N_kpt, dtype=complex)
    kj = k[j, :2]

    for r1 in range(nr):
        phase_r1 = np.exp(1j * np.sum(k[:, :2] * R_mNN[r1, :2], axis=1))
        cc_r = Cc[:, :, c0] * phase_r1[:, None]

        for r2 in range(nr):
            chi_r1r2 = chi_p[:, :, :, :, r1, r2]
            s1 = np.sum(np.conj(Cv[j, :, vv0])[:, None, None, None] * chi_r1r2, axis=0)

            phase_r2 = np.exp(1j * np.sum(kj * R_mNN[r2, :2]))
            cc_rr = Cc[j, :, cc0] * phase_r2
            s2 = np.sum(cc_rr[:, None, None] * s1, axis=0)

            vx += np.sum((Cv[:, :, v0] @ s2) * np.conj(cc_r), axis=1)

    return vx


# ==========================================================================
# EXCHANGE_COULOMB_INTERACTION_BATCHED_J_PY
#
# DESCRIPTION:
#   Computes one ordered source-j chunk for a single target band block.
#
# INPUT ARGUMENTS:
#   chi_p / chi_packed_descriptor : precomputed chi tensor or its read-only
#       packed-memmap descriptor; only one is required.
#   j_indices : ordered source k-point indices, shape (n_chunk,).
#   Cv, Cc    : complex band coefficients with runtime Wannier dimension NB.
#
# OUTPUT:
#   vx : complex128 exchange block, shape (N_kpt, n_chunk).
#
# METHODOLOGY:
#   Uses the validated S1 matrix product when packed chi is supplied:
#   S1_bcde = sum_a Cv*_ba chi_acde.  The single-target target reduction is
#   intentionally retained as the established fallback implementation.
#
# NOTES:
#   Output columns preserve j_indices exactly.  Band indices remain 1-based
#   at this interface; no phase, conjugation, or R1/R2 ordering is changed.
# ==========================================================================
def exchange_coulomb_interaction_batched_j_py(
    chi_p, R_mNN, k, N_kpt, Cv, Cc, c, cc, v, vv, j_indices, profile=None,
    chi_packed_descriptor=None
):
    """
    exchange_coulomb_interaction_batched_j_py
    
    PURPOSE:
        Contract exchange columns for an explicitly ordered source-j chunk.
    
    DESCRIPTION:
        ``j_indices`` is never reordered. The returned complex128 array has
        shape ``(N_kpt, len(j_indices))`` and column ``p`` corresponds exactly to
        ``j_indices[p]``. This is a production-capable helper only; the current
        one-j HBSE joblib scheduling continues to call
        :func:`exchange_coulomb_interaction_py`.
    
    INPUT ARGUMENTS:
        See the annotated signature and detailed description.
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    total_start = time.perf_counter() if profile is not None else None
    R_mNN = np.asarray(R_mNN, dtype=float)
    k = np.asarray(k, dtype=float)
    Cv = np.asarray(Cv)
    Cc = np.asarray(Cc)
    js = np.asarray(j_indices, dtype=np.intp)

    if js.ndim != 1 or js.size == 0:
        raise ValueError("j_indices must be a nonempty one-dimensional sequence")
    if np.any(js < 0) or np.any(js >= N_kpt):
        raise ValueError("j_indices contain an out-of-range k-point")
    if R_mNN.ndim != 2 or R_mNN.shape[1] < 2:
        raise ValueError(f"R_mNN must have shape (NR, >=2), got {R_mNN.shape}")
    if chi_packed_descriptor is None:
        chi_p = np.asarray(chi_p)
        if chi_p.ndim != 6:
            raise ValueError(f"chi_p must be rank-6, got shape {chi_p.shape}")
        if chi_p.shape[4] != R_mNN.shape[0] or chi_p.shape[5] != R_mNN.shape[0]:
            raise ValueError(f"chi_p/R_mNN mismatch: chi_p shape {chi_p.shape}, R_mNN shape {R_mNN.shape}")
    if k.shape[0] != N_kpt:
        raise ValueError(f"N_kpt={N_kpt} does not match k shape {k.shape}")

    c0, cc0, v0, vv0 = c - 1, cc - 1, v - 1, vv - 1
    nr = R_mNN.shape[0]
    n_chunk = js.size
    packed_chi, packed_map_created, packed_map_open_s = _get_packed_chi_memmap(
        chi_packed_descriptor, Cv.shape[1], nr
    )
    vx = np.zeros((N_kpt, n_chunk), dtype=np.complex128)
    source_s1_s = 0.0
    source_s2_s = 0.0
    target_matmul_s = 0.0
    target_reduce_s = 0.0
    cv_source = np.conj(Cv[js, :, vv0])
    # Stage 4 validated: this target-band view is non-contiguous because the
    # band axis is interleaved. The copy is chunk-local and reused by all
    # R1/R2 contractions without changing arithmetic or dtype.
    cv_target = np.ascontiguousarray(Cv[:, :, v0])

    for r1 in range(nr):
        phase_r1 = np.exp(1j * np.sum(k[:, :2] * R_mNN[r1, :2], axis=1))
        cc_r = Cc[:, :, c0] * phase_r1[:, None]
        # Depends on r1 and target conduction band c only; reuse for all r2.
        conj_cc_r = np.conj(cc_r)

        for r2 in range(nr):
            # ba,acde->bcde contracts the first chi basis axis for every j.
            op_start = time.perf_counter() if profile is not None else None
            if packed_chi is None:
                chi_r1r2 = chi_p[:, :, :, :, r1, r2]
                s1 = np.einsum("ba,acde->bcde", cv_source, chi_r1r2, optimize=True)
            else:
                s1 = (cv_source @ packed_chi[r1 * nr + r2]).reshape(
                    n_chunk, Cv.shape[1], Cv.shape[1], Cv.shape[1]
                )
            if op_start is not None:
                source_s1_s += time.perf_counter() - op_start

            phase_r2 = np.exp(1j * np.sum(k[js, :2] * R_mNN[r2, :2], axis=1))
            cc_rr = Cc[js, :, cc0] * phase_r2[:, None]
            # ba,bacd->bcd contracts the second chi basis axis for every j.
            op_start = time.perf_counter() if profile is not None else None
            s2 = np.einsum("ba,bacd->bcd", cc_rr, s1, optimize=True)
            if op_start is not None:
                source_s2_s += time.perf_counter() - op_start

            # B independent target-k contractions; transpose restores (k, j).
            op_start = time.perf_counter() if profile is not None else None
            target_by_source = np.matmul(cv_target[None, :, :], s2)
            if op_start is not None:
                target_matmul_s += time.perf_counter() - op_start
            op_start = time.perf_counter() if profile is not None else None
            vx += np.sum(target_by_source * conj_cc_r[None, :, :], axis=2).T
            if op_start is not None:
                target_reduce_s += time.perf_counter() - op_start

    if profile is not None:
        kernel_s = time.perf_counter() - total_start
        profile["calls"] = profile.get("calls", 0) + 1
        profile["r_pairs"] = profile.get("r_pairs", 0) + nr * nr
        profile["target_blocks"] = profile.get("target_blocks", 0) + 1
        profile["j_columns"] = profile.get("j_columns", 0) + n_chunk
        profile["batched_kernel_s"] = profile.get("batched_kernel_s", 0.0) + kernel_s
        profile["batched_source_s1_s"] = profile.get("batched_source_s1_s", 0.0) + source_s1_s
        profile["batched_source_s2_s"] = profile.get("batched_source_s2_s", 0.0) + source_s2_s
        profile["batched_target_matmul_s"] = profile.get("batched_target_matmul_s", 0.0) + target_matmul_s
        profile["batched_target_reduce_s"] = profile.get("batched_target_reduce_s", 0.0) + target_reduce_s
        profile["batched_other_s"] = profile.get("batched_other_s", 0.0) + max(
            0.0, kernel_s - source_s1_s - source_s2_s - target_matmul_s - target_reduce_s
        )
        profile["packed_chi_map_creations"] = profile.get("packed_chi_map_creations", 0) + int(packed_map_created)
        profile["packed_chi_map_open_s"] = profile.get("packed_chi_map_open_s", 0.0) + packed_map_open_s

    return vx


# ==========================================================================
# EXCHANGE_COULOMB_INTERACTION_GROUPED_SOURCE_PAIR_PY
#
# DESCRIPTION:
#   Computes exchange blocks for one source band pair and all of its target
#   band blocks for one ordered source-j chunk.
#
# INPUT ARGUMENTS:
#   chi_p / chi_packed_descriptor : precomputed chi tensor or parent-created
#       read-only packed-memmap descriptor.
#   target_blocks : target band metadata with 1-based v/c indices.
#   j_indices     : ordered source k-point indices, shape (n_chunk,).
#
# OUTPUT:
#   outputs : target-indexed complex128 Vx blocks, each (N_kpt, n_chunk).
#
# METHODOLOGY:
#   S1 is evaluated as a packed matrix multiplication preserving
#   S1_bcde = sum_a Cv*_ba chi_acde.  The target reduction is fused as
#   einsum("jkd,kd->kj") into one reusable (N_kpt, n_chunk) buffer, avoiding
#   the former explicit (n_chunk, N_kpt, NB) scratch product.  R1/R2 order,
#   S2 reuse, phases, conjugation, and accumulation order are unchanged.
#
# NOTES:
#   Workers only map packed chi read-only and cache it per process.  The
#   single-target fallback deliberately retains its established reduction.
# ==========================================================================
def exchange_coulomb_interaction_grouped_source_pair_py(
    chi_p, R_mNN, k, N_kpt, Cv, Cc, vv, cc, target_blocks, j_indices, profile=None,
    chi_packed_descriptor=None
):
    """
    exchange_coulomb_interaction_grouped_source_pair_py
    
    PURPOSE:
        Exchange coulomb interaction grouped source pair py.
    
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
    if len(target_blocks) == 1:
        block = target_blocks[0]
        return {
            0: exchange_coulomb_interaction_batched_j_py(
                chi_p, R_mNN, k, N_kpt, Cv, Cc,
                block["c"], cc, block["v"], vv, j_indices, profile=profile,
                chi_packed_descriptor=chi_packed_descriptor
            )
        }

    R_mNN = np.asarray(R_mNN, dtype=float)
    k = np.asarray(k, dtype=float)
    Cv = np.asarray(Cv)
    Cc = np.asarray(Cc)
    js = np.asarray(j_indices, dtype=np.intp)

    if js.ndim != 1 or js.size == 0:
        raise ValueError("j_indices must be a nonempty one-dimensional sequence")
    if np.any(js < 0) or np.any(js >= N_kpt):
        raise ValueError("j_indices contain an out-of-range k-point")
    if R_mNN.ndim != 2 or R_mNN.shape[1] < 2:
        raise ValueError(f"R_mNN must have shape (NR, >=2), got {R_mNN.shape}")
    if chi_packed_descriptor is None:
        chi_p = np.asarray(chi_p)
        if chi_p.ndim != 6:
            raise ValueError(f"chi_p must be rank-6, got shape {chi_p.shape}")
        if chi_p.shape[4] != R_mNN.shape[0] or chi_p.shape[5] != R_mNN.shape[0]:
            raise ValueError(f"chi_p/R_mNN mismatch: chi_p shape {chi_p.shape}, R_mNN shape {R_mNN.shape}")
    if k.shape[0] != N_kpt:
        raise ValueError(f"N_kpt={N_kpt} does not match k shape {k.shape}")

    vv0 = vv - 1
    cc0 = cc - 1
    nr = R_mNN.shape[0]
    n_chunk = js.size
    packed_chi, packed_map_created, packed_map_open_s = _get_packed_chi_memmap(
        chi_packed_descriptor, Cv.shape[1], nr
    )

    total_start = time.perf_counter() if profile is not None else None
    cv_source = np.conj(Cv[js, :, vv0])
    cc_source = Cc[js, :, cc0]
    cc_rr_cache = []
    for r2 in range(nr):
        phase_r2 = np.exp(1j * np.sum(k[js, :2] * R_mNN[r2, :2], axis=1))
        cc_rr_cache.append(cc_source * phase_r2[:, None])

    s2_cache = {}
    source_start = time.perf_counter() if profile is not None else None
    source_s1_s = 0.0
    source_s2_s = 0.0
    for r1 in range(nr):
        for r2 in range(nr):
            op_start = time.perf_counter() if profile is not None else None
            if packed_chi is None:
                chi_r1r2 = chi_p[:, :, :, :, r1, r2]
                s1 = np.einsum("ba,acde->bcde", cv_source, chi_r1r2, optimize=True)
            else:
                s1 = (cv_source @ packed_chi[r1 * nr + r2]).reshape(
                    n_chunk, Cv.shape[1], Cv.shape[1], Cv.shape[1]
                )
            if op_start is not None:
                source_s1_s += time.perf_counter() - op_start
            op_start = time.perf_counter() if profile is not None else None
            s2_cache[(r1, r2)] = np.einsum("ba,bacd->bcd", cc_rr_cache[r2], s1, optimize=True)
            if op_start is not None:
                source_s2_s += time.perf_counter() - op_start

    reduction_buffer = np.empty((N_kpt, n_chunk), dtype=np.complex128)
    outputs = {}
    target_start = time.perf_counter() if profile is not None else None
    target_matmul_s = 0.0
    target_reduce_s = 0.0
    for target_index, block in enumerate(target_blocks):
        v0 = block["v"] - 1
        c0 = block["c"] - 1
        vx = np.zeros((N_kpt, n_chunk), dtype=np.complex128)
        cv_target = np.ascontiguousarray(Cv[:, :, v0])
        for r1 in range(nr):
            phase_r1 = np.exp(1j * np.sum(k[:, :2] * R_mNN[r1, :2], axis=1))
            cc_r = Cc[:, :, c0] * phase_r1[:, None]
            conj_cc_r = np.conj(cc_r)
            for r2 in range(nr):
                op_start = time.perf_counter() if profile is not None else None
                target_by_source = np.matmul(cv_target[None, :, :], s2_cache[(r1, r2)])
                if op_start is not None:
                    target_matmul_s += time.perf_counter() - op_start
                op_start = time.perf_counter() if profile is not None else None
                # Equivalent to the previous multiply-plus-sum, but fuses the
                # d-axis reduction and reuses the smaller output buffer.
                np.einsum(
                    "jkd,kd->kj", target_by_source, conj_cc_r,
                    out=reduction_buffer, optimize=True,
                )
                vx += reduction_buffer
                if op_start is not None:
                    target_reduce_s += time.perf_counter() - op_start
        outputs[target_index] = vx

    if profile is not None:
        source_phase_s = target_start - source_start
        target_phase_s = time.perf_counter() - target_start
        kernel_s = time.perf_counter() - total_start
        profile["calls"] = profile.get("calls", 0) + 1
        profile["r_pairs"] = profile.get("r_pairs", 0) + nr * nr
        profile["target_blocks"] = profile.get("target_blocks", 0) + len(target_blocks)
        profile["j_columns"] = profile.get("j_columns", 0) + n_chunk * len(target_blocks)
        profile["source_phase_s"] = profile.get("source_phase_s", 0.0) + source_phase_s
        profile["target_phase_s"] = profile.get("target_phase_s", 0.0) + target_phase_s
        profile["kernel_s"] = profile.get("kernel_s", 0.0) + kernel_s
        profile["grouped_source_s1_s"] = profile.get("grouped_source_s1_s", 0.0) + source_s1_s
        profile["grouped_source_s2_s"] = profile.get("grouped_source_s2_s", 0.0) + source_s2_s
        profile["grouped_source_other_s"] = profile.get("grouped_source_other_s", 0.0) + max(
            0.0, source_phase_s - source_s1_s - source_s2_s
        )
        profile["grouped_target_matmul_s"] = profile.get("grouped_target_matmul_s", 0.0) + target_matmul_s
        profile["grouped_target_reduce_s"] = profile.get("grouped_target_reduce_s", 0.0) + target_reduce_s
        profile["grouped_target_other_s"] = profile.get("grouped_target_other_s", 0.0) + max(
            0.0, target_phase_s - target_matmul_s - target_reduce_s
        )
        profile["packed_chi_map_creations"] = profile.get("packed_chi_map_creations", 0) + int(packed_map_created)
        profile["packed_chi_map_open_s"] = profile.get("packed_chi_map_open_s", 0.0) + packed_map_open_s

    return outputs
