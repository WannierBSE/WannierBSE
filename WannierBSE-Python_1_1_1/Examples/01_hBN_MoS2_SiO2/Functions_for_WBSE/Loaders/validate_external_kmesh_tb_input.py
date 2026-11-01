"""
validate_external_kmesh_tb_input

PURPOSE:
    Validate user-provided kmesh/TB_data inputs.

DESCRIPTION:
    For now, User_input/kmesh.txt may be provided without User_input/TB_data.
    When the external TB set is incomplete, the caller should treat the kmesh as
    the active mesh and regenerate internal TB data for that mesh.

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
from pathlib import Path


@dataclass(frozen=True)
class ExternalTBInput:
    """
    ExternalTBInput
    
    PURPOSE:
        Configuration or data container used by validate_external_kmesh_tb_input.
    
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
    use_user_provided_kmesh_tb: bool
    user_kmesh_path: Path
    user_bands_path: Path
    has_user_kmesh: bool
    has_user_tb_folder: bool
    missing_band_files: tuple[str, ...]


def _required_band_files(nv: int, nc: int) -> list[str]:
    """
    _required_band_files
    
    PURPOSE:
        Required band files.
    
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
    return [f"v{i}_TB.txt" for i in range(1, nv + 1)] + [f"c{i}_TB.txt" for i in range(1, nc + 1)]


def _missing_band_files(user_bands_path: Path, nv: int, nc: int) -> tuple[str, ...]:
    """
    _missing_band_files
    
    PURPOSE:
        Missing band files.
    
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
    required = _required_band_files(nv, nc)
    if not user_bands_path.is_dir():
        return tuple(required)
    return tuple(name for name in required if not (user_bands_path / name).is_file())


def validate_external_kmesh_tb_input_py(
    user_input_path: str | Path,
    nv: int,
    nc: int,
    exchange_interaction: bool,
) -> ExternalTBInput:
    """
    validate_external_kmesh_tb_input_py
    
    PURPOSE:
        Validate external kmesh tb input py.
    
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
    user_input_path = Path(user_input_path)
    user_kmesh_path = user_input_path / "kmesh.txt"
    user_bands_path = user_input_path / "TB_data"

    has_user_kmesh = user_kmesh_path.is_file()
    has_user_tb_folder = user_bands_path.is_dir()
    missing = _missing_band_files(user_bands_path, nv, nc)
    has_complete_user_tb = not missing
    use_external = has_user_kmesh and has_complete_user_tb

    if not has_user_kmesh and has_user_tb_folder:
        raise RuntimeError(
            "[ERROR] User TB_data requires a matching user kmesh. "
            "Provide User_input/kmesh.txt or remove User_input/TB_data."
        )

    if use_external and exchange_interaction:
        raise RuntimeError(
            "User-provided kmesh/TB_data is supported only for direct-interaction calculations. "
            "Disable Exchange_Interaction or use internally generated Wannier inputs."
        )

    return ExternalTBInput(
        use_user_provided_kmesh_tb=use_external,
        user_kmesh_path=user_kmesh_path,
        user_bands_path=user_bands_path,
        has_user_kmesh=has_user_kmesh,
        has_user_tb_folder=has_user_tb_folder,
        missing_band_files=missing,
    )
