"""
read_exchange_control

PURPOSE:
    Read MATLAB-style exchange-control parameters for WannierBSE.

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

import re
from pathlib import Path

import numpy as np


_DEFAULTS = {
    "Exchange_Interaction": False,
    "Use_Precomputed_Chi": False,
    "cleanup_xsf": False,
    "M_max": 1,
    "G_max": 4.2,
    "tol_probability": 0.98,
    "M_max_tolerance": 0.01,
    "nn_distance_tolerance": 0.15,
}


def _parse_numeric_or_text(text: str):
    """
    _parse_numeric_or_text
    
    PURPOSE:
        Parse numeric or text.
    
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
    value = text.strip().rstrip(";").strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"true", "false", "yes", "no", "on", "off"}:
        return lowered in {"true", "yes", "on"}
    try:
        numeric = np.fromstring(value, sep=" ")
        if numeric.size == 1:
            return numeric.item()
        if numeric.size > 1:
            return numeric
    except ValueError:
        pass
    return value


def parse_cleanup_xsf_py(value) -> bool:
    """
    parse_cleanup_xsf_py
    
    PURPOSE:
        Strict cleanup_xsf parser: only trimmed, case-insensitive 'true' enables cleanup.
    
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
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def read_exchange_control_py(path: str | Path) -> dict:
    """
    read_exchange_control_py
    
    PURPOSE:
        Read ``exchange_control.txt`` with MATLAB ``read_params.m`` key behavior.
    
    DESCRIPTION:
        MATLAB accepts parameter names containing spaces and normalizes whitespace
        to underscores, so ``M_max tolerance`` becomes ``M_max_tolerance``.
        Deprecated lowercase aliases are also mirrored for compatibility.
    
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
    raw_params = {}
    params = dict(_DEFAULTS)
    if not path.is_file():
        return params

    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_ ]*)\s*=\s*(.*?)\s*$")
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("%", 1)[0].strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            continue
        key = re.sub(r"\s+", "_", match.group(1).strip())
        if key == "cleanup_xsf":
            raw_params[key] = parse_cleanup_xsf_py(match.group(2))
        else:
            raw_params[key] = _parse_numeric_or_text(match.group(2))

    params.update(raw_params)
    aliases = {
        "m_max": "M_max",
        "g_max": "G_max",
        "m_max_tolerance": "M_max_tolerance",
    }
    for old_key, new_key in aliases.items():
        if old_key in raw_params and new_key not in raw_params:
            params[new_key] = raw_params[old_key]

    return params
