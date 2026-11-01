"""Read-only comparison of cached and configured k-mesh controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .read_params import read_params_py


def _integer_scalar(value) -> bool:
    """Return whether value is one finite integer-valued scalar."""
    try:
        array = np.asarray(value)
        if array.size != 1:
            return False
        number = float(array.reshape(-1)[0])
        return np.isfinite(number) and number.is_integer()
    except (TypeError, ValueError, OverflowError):
        return False


def cached_kmesh_config_mismatch(cache_file: str | Path,
                                 control_file: str | Path) -> bool:
    """Compare the three control fields represented by a complete cache header.

    Cache-header order is ``Dense_range, Nk_coarse, Nk_dense, total_coarse,
    total_dense``.  The derived totals are validated for completeness but are
    not compared with control values.  Missing, malformed, or incomplete
    inputs return ``False`` because a mismatch cannot be confirmed reliably.
    """
    cache_file = Path(cache_file)
    control_file = Path(control_file)
    if not cache_file.is_file() or not control_file.is_file():
        return False

    try:
        params = read_params_py(control_file)
        names = ('Nk_coarse', 'Nk_dense', 'Dense_range')
        if any(name not in params or not _integer_scalar(params[name]) for name in names):
            return False
        control_values = [int(float(np.asarray(params[name]).reshape(-1)[0])) for name in names]

        header = []
        with cache_file.open('r') as stream:
            for _ in range(5):
                line = stream.readline()
                if not line:
                    return False
                value = float(line.strip())
                if not np.isfinite(value) or not value.is_integer():
                    return False
                header.append(int(value))

        cached_values = [header[1], header[2], header[0]]
        return cached_values != control_values
    except (OSError, ValueError, TypeError, OverflowError, KeyError):
        return False
