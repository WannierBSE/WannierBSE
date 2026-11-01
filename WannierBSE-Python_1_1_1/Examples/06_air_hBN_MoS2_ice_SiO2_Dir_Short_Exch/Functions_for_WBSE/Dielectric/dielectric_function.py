"""
dielectric_function

PURPOSE:
    Layered quasi-2D dielectric solver translated from MATLAB dielectric_function.m.

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

from Functions_for_WBSE.Loaders.read_params import read_params_py


def _parse_control_file(path: str | Path) -> dict[str, np.ndarray | float | int]:
    """
    _parse_control_file
    
    PURPOSE:
        Parse control file.
    
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
    params = {}
    for line in Path(path).read_text().splitlines():
        clean = line.split("%", 1)[0].strip()
        if not clean or "=" not in clean:
            continue
        key, value = clean.split("=", 1)
        key = key.strip()
        tokens = value.replace(";", "").replace(",", " ").split()
        numbers = np.array([float(tok.replace("D", "E").replace("d", "e")) for tok in tokens], dtype=float)
        if numbers.size == 1:
            params[key] = numbers.item()
        else:
            params[key] = numbers
    return params


def _load_physical_constants(path: str | Path) -> dict[str, float]:
    """
    _load_physical_constants
    
    PURPOSE:
        Load physical constants.
    
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
    params = read_params_py(path)
    required = ("e", "m0", "hbar")
    missing = [key for key in required if key not in params]
    if missing:
        raise KeyError(f"Missing physical constants in {path}: {', '.join(missing)}")
    return {key: float(params[key]) for key in params}


def dielectric_function_py(
    control_file: str | Path,
    physical_constants_file: str | Path,
    output_file_path: str | Path,
    q_max_input: float = 4.0,
    num_q: int = 5000,
    num_z: int = 100,
) -> np.ndarray:
    """
    dielectric_function_py
    
    PURPOSE:
        Generate epsilon(q) and write the WBSE two-column dielectric table.
    
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
    params = _parse_control_file(control_file)
    phy = _load_physical_constants(physical_constants_file)

    # MATLAB dielectric_function.m intentionally assigns s=t_layers and
    # t=s_layers; preserve that legacy indexing so boundary rows match.
    s_layers = int(params["t_layers"])
    t_layers = int(params["s_layers"])
    t_dielectric = np.asarray(params["t_dielectric"], dtype=float).reshape(-1)
    s_dielectric = np.asarray(params["s_dielectric"], dtype=float).reshape(-1)
    if t_dielectric.size != s_layers:
        raise ValueError(f"Length of t_dielectric ({t_dielectric.size}) must match t_layers ({s_layers}).")
    if s_dielectric.size != t_layers:
        raise ValueError(f"Length of s_dielectric ({s_dielectric.size}) must match s_layers ({t_layers}).")

    e_layers = np.concatenate([t_dielectric, np.array([np.nan]), s_dielectric])
    e_si = phy["e"]
    m0_si = phy["m0"]
    _hbar_si = phy["hbar"]

    ang_to_nm = 0.1
    conv_factor_e = (e_si / m0_si) * 1.0e-6
    z_legacy = np.asarray(params["layer_boundaries"], dtype=float).reshape(-1) * ang_to_nm
    q_max_legacy = q_max_input / ang_to_nm
    epl_legacy = float(params["Epl"]) * conv_factor_e
    alpha = float(params["alpha"])
    ep_p = float(params["ep_p"])

    leg_epsilon0 = 3.1426e-7
    leg_e = 1.0
    leg_m0 = 1.0
    leg_hbar = 1.1582e2

    n_layers = s_layers + 1 + t_layers
    if z_legacy.size != n_layers - 1:
        raise ValueError("layer_boundaries must contain t_layers + s_layers values.")

    term1 = (leg_e**2 * leg_m0) / (np.pi**2 * leg_epsilon0 * leg_hbar**2)
    term2 = (3.0 * np.pi**2 * leg_epsilon0 * leg_m0 * epl_legacy**2) / (leg_e**2 * leg_hbar**2)
    q_tf = np.sqrt(term1 * term2 ** (1.0 / 3.0))
    h = z_legacy[s_layers - 1] - z_legacy[s_layers]
    dz = h / float(num_z)
    idx_vec = np.arange(1, num_z + 1, dtype=float)
    z_grid = -h / 2.0 + idx_vec * dz
    zp_grid = z_grid.reshape(1, -1)
    dist_matrix = np.abs(z_grid.reshape(-1, 1) - zp_grid)

    q_vec_legacy = np.zeros(num_q, dtype=float)
    etilde_out = np.zeros(num_q, dtype=float)

    for i in range(num_q):
        q = (i + 1) * (q_max_legacy / float(num_q))
        q_vec_legacy[i] = q
        term_tmd_1 = 1.0 / (ep_p - 1.0)
        term_tmd_2 = alpha * q**2 / q_tf**2
        term_tmd_3 = (leg_hbar**2 * q**2 / (2.0 * leg_m0 * epl_legacy)) ** 2
        tmd_val = 1.0 + (term_tmd_1 + term_tmd_2 + term_tmd_3) ** (-1.0)
        e_local = e_layers.copy()
        e_local[s_layers] = tmd_val

        a_mat = np.zeros((2 * n_layers - 2, 2 * n_layers - 2), dtype=float)
        a_mat[0, 0] = np.exp(-q * z_legacy[0]) / e_local[0]
        a_mat[0, 1] = -np.exp(-q * z_legacy[0]) / e_local[1]
        a_mat[0, 2] = -np.exp(q * z_legacy[0]) / e_local[1]
        a_mat[1, 0] = -np.exp(-q * z_legacy[0])
        a_mat[1, 1] = np.exp(-q * z_legacy[0])
        a_mat[1, 2] = -np.exp(q * z_legacy[0])

        if s_layers > 1:
            for j in range(1, s_layers):
                row1 = 2 * j
                row2 = 2 * j + 1
                z = z_legacy[j]
                a_mat[row1, 2*j - 1] = np.exp(-q * z) / e_local[j]
                a_mat[row1, 2*j] = np.exp(q * z) / e_local[j]
                a_mat[row1, 2*j + 1] = -np.exp(-q * z) / e_local[j + 1]
                a_mat[row1, 2*j + 2] = -np.exp(q * z) / e_local[j + 1]
                a_mat[row2, 2*j - 1] = -np.exp(-q * z)
                a_mat[row2, 2*j] = np.exp(q * z)
                a_mat[row2, 2*j + 1] = np.exp(-q * z)
                a_mat[row2, 2*j + 2] = -np.exp(q * z)

        if t_layers > 1:
            for k_idx in range(1, t_layers):
                row = 2 * s_layers + 2 * k_idx - 2
                z = z_legacy[s_layers + k_idx - 1]
                a_mat[row, 2*s_layers + 2*k_idx - 3] = -np.exp(-q * z) / e_local[s_layers + k_idx - 1]
                a_mat[row, 2*s_layers + 2*k_idx - 2] = -np.exp(q * z) / e_local[s_layers + k_idx - 1]
                a_mat[row, 2*s_layers + 2*k_idx - 1] = np.exp(-q * z) / e_local[s_layers + k_idx]
                a_mat[row, 2*s_layers + 2*k_idx] = np.exp(q * z) / e_local[s_layers + k_idx]
                a_mat[row + 1, 2*s_layers + 2*k_idx - 3] = np.exp(-q * z)
                a_mat[row + 1, 2*s_layers + 2*k_idx - 2] = -np.exp(q * z)
                a_mat[row + 1, 2*s_layers + 2*k_idx - 1] = -np.exp(-q * z)
                a_mat[row + 1, 2*s_layers + 2*k_idx] = np.exp(q * z)

        z_last = z_legacy[n_layers - 2]
        a_mat[2*n_layers - 4, 2*n_layers - 5] = -np.exp(-q * z_last) / e_local[n_layers - 2]
        a_mat[2*n_layers - 4, 2*n_layers - 4] = -np.exp(q * z_last) / e_local[n_layers - 2]
        a_mat[2*n_layers - 4, 2*n_layers - 3] = np.exp(q * z_last) / e_local[n_layers - 1]
        a_mat[2*n_layers - 3, 2*n_layers - 5] = np.exp(-q * z_last)
        a_mat[2*n_layers - 3, 2*n_layers - 4] = -np.exp(q * z_last)
        a_mat[2*n_layers - 3, 2*n_layers - 3] = np.exp(q * z_last)

        b_all = np.zeros((2 * n_layers - 2, num_z), dtype=float)
        b_all[2*s_layers - 2, :] = np.exp(-q * (z_legacy[s_layers - 1] - zp_grid)) / tmd_val
        b_all[2*s_layers - 1, :] = -np.exp(-q * (z_legacy[s_layers - 1] - zp_grid))
        b_all[2*s_layers, :] = np.exp(-q * (zp_grid - z_legacy[s_layers])) / tmd_val
        b_all[2*s_layers + 1, :] = np.exp(-q * (zp_grid - z_legacy[s_layers]))

        scale = np.max(np.abs(a_mat), axis=1)
        scale[scale == 0] = 1.0
        a_scaled = a_mat / scale[:, None]
        b_scaled = b_all / scale[:, None]
        try:
            x_all = np.linalg.solve(a_scaled, b_scaled)
        except np.linalg.LinAlgError:
            x_all = np.linalg.lstsq(a_scaled, b_scaled, rcond=None)[0]
        a0_vec = x_all[2*s_layers - 1, :]
        b0_vec = x_all[2*s_layers, :]

        v_pref = leg_e**2 / (2.0 * leg_epsilon0 * q)
        w_pref = leg_e**2 / (2.0 * tmd_val * leg_epsilon0 * q)
        vtilde = np.sum(np.exp(-q * dist_matrix)) * v_pref * dz**2
        kernel = np.exp(-q * z_grid).reshape(-1, 1) * a0_vec.reshape(1, -1)
        kernel += np.exp(q * z_grid).reshape(-1, 1) * b0_vec.reshape(1, -1)
        kernel += np.exp(-q * dist_matrix)
        wtilde = np.sum(kernel) * w_pref * dz**2
        etilde_out[i] = (vtilde / h**2) / (wtilde / h**2)

    q_out_si = q_vec_legacy * ang_to_nm
    q_final = np.concatenate([[0.0], q_out_si])
    e_final = np.concatenate([[(e_local[0] + e_local[-1]) / 2.0], etilde_out])
    output = np.column_stack([q_final, e_final])
    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_file_path, output, fmt="%.15g", delimiter="\t")
    return output
