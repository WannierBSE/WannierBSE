"""
kmesh_band_sync

PURPOSE:
    Synchronize external TB files with the sorted WBSE kmesh order.

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


def kmesh_band_sync_py(
    source_path: str | Path,
    dest_path: str | Path,
    index_map: np.ndarray,
    nv: int,
    nc: int,
    nb: int,
    k_fixed: np.ndarray,
) -> None:
    """
    kmesh_band_sync_py
    
    PURPOSE:
        Kmesh band sync py.
    
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
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    band_files = [f"v{i}_TB.txt" for i in range(1, nv + 1)] + [f"c{i}_TB.txt" for i in range(1, nc + 1)]
    n_kpt = len(index_map)
    lines_per_block = 2 + int(nb)
    k_fixed = np.asarray(k_fixed, dtype=float)

    for file_name in band_files:
        input_path = source_path / file_name
        output_path = dest_path / file_name.replace("_TB.txt", "_TB_WBSE.txt")
        if not input_path.is_file():
            raise FileNotFoundError(f"Band file not found: {input_path}")

        lines = input_path.read_text().splitlines()
        expected_min = n_kpt * lines_per_block
        if len(lines) < expected_min:
            raise ValueError(
                f"File {input_path} has {len(lines)} lines; expected at least "
                f"{expected_min} for {n_kpt} k-points and NB={nb}."
            )

        sorted_lines: list[str] = []
        for target_idx, original_idx in enumerate(index_map):
            source_start = int(original_idx) * lines_per_block
            kx, ky = k_fixed[target_idx, 0], k_fixed[target_idx, 1]
            sorted_lines.append(f"{kx:20.16f}    {ky:20.16f}     {0.0:20.16f}")
            sorted_lines.extend(lines[source_start + 1 : source_start + lines_per_block])

        output_path.write_text("\n".join(sorted_lines) + "\n")
