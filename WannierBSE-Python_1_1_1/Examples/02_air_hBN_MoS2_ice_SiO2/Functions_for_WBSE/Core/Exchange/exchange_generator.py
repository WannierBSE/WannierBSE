"""
exchange_generator

PURPOSE:
    Stage 6B exchange tensor generator orchestration.

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
import time

import h5py
import numpy as np

from Functions_for_WBSE.Core.Exchange.array_utils import matlab_complex_to_numpy, write_matlab_dataset
from Functions_for_WBSE.Core.Exchange.chi_calculation import chi_calculation_py
from Functions_for_WBSE.Core.Exchange.coordinate_transformation import coordinate_transformation_py
from Functions_for_WBSE.Core.Exchange.find_rgrid_in_mmn import find_rgrid_in_mmn_py
from Functions_for_WBSE.Core.Exchange.find_truncated_wfs import find_truncated_wfs_py
from Functions_for_WBSE.Core.Exchange.generate_g_vectors import generate_g_vectors_py
from Functions_for_WBSE.Core.Exchange.read_grid_info import read_grid_info_py


def load_wannier_cache_py(wf_path):
    """
    load_wannier_cache_py
    
    PURPOSE:
        Load wannier cache py.
    
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
    wf_path = Path(wf_path)
    with h5py.File(wf_path / "WF_up.mat", "r") as h5:
        W_up = matlab_complex_to_numpy(h5["W_up"][()]).T
    with h5py.File(wf_path / "WF_down.mat", "r") as h5:
        W_down = matlab_complex_to_numpy(h5["W_down"][()]).T
    with h5py.File(wf_path / "r.mat", "r") as h5:
        r = np.asarray(h5["r"][()]).T
    return W_up, W_down, r


def _complex_mib(shape):
    """
    _complex_mib
    
    PURPOSE:
        Complex mib.
    
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
    return float(np.prod(shape) * np.dtype(complex).itemsize / 1024**2)


def generate_exchange_tensors_py(
    root_path,
    a1,
    a2,
    tau,
    n_atoms,
    basis_tau_index,
    b1,
    b2,
    e,
    eps0,
    params,
):
    """
    generate_exchange_tensors_py
    
    PURPOSE:
        Generate exchange tensors py.
    
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
    start = time.perf_counter()
    root = Path(root_path)
    wf_path = root / "Precomputed_data" / "Wannier_functions_WBSE"
    out_cache = root / "Precomputed_data" / "Exchange_interaction"
    out_data = root / "Exciton_data" / "Exchange_data"
    W_up, W_down, r = load_wannier_cache_py(wf_path)
    print(
        f"[INFO ] Exchange generator: loaded W_up={W_up.shape}, W_down={W_down.shape}, r={r.shape}",
        flush=True,
    )
    na1, na2, na3, delta_v, supercell_size, shift_a1, shift_a2, shift_a3 = read_grid_info_py(wf_path, a1)
    jac, hu, hv, hz, u, v, u_point, v_point, z_point = coordinate_transformation_py(r, a1, a2, na3)
    R_mNN, n1_n2_mNN, d_max = find_rgrid_in_mmn_py(a1, a2, tau, n_atoms, supercell_size, int(params["M_max"]), float(params["nn_distance_tolerance"]), float(params["M_max_tolerance"]))
    trunc = find_truncated_wfs_py(W_up, W_down, r, u, v, delta_v, na1, z_point, float(params["tol_probability"]))
    W_up_t, W_down_t, tin_all, r_t, u_t, v_t, z_t, u_point_t, v_point_t, z_point_t, nu_t, nv_t, nz_t, na1_t, na2_t, zcut_min, zcut_max = trunc
    kex = np.array([0.0, 0.0, 0.0])
    G_all = generate_g_vectors_py(float(params["G_max"]), b1, b2)
    nb = W_up_t.shape[1]
    rho_shape = (nb, nb, int(zcut_max - zcut_min + 1), len(R_mNN), len(G_all))
    chi_shape = (nb, nb, nb, nb, len(R_mNN), len(R_mNN))
    print(
        f"[INFO ] Exchange generator: R_mNN={len(R_mNN)}, G_all={len(G_all)}, "
        f"truncated_points={len(tin_all)}, Nz_t={int(nz_t)}",
        flush=True,
    )
    print(
        f"[INFO ] Exchange generator: rho_shape={rho_shape} (~{_complex_mib(rho_shape):.1f} MiB), "
        f"chi_shape={chi_shape} (~{_complex_mib(chi_shape):.1f} MiB)",
        flush=True,
    )
    from Functions_for_WBSE.Core.Exchange.rho_calculation import rho_calculation_py
    rho_start = time.perf_counter()
    rho = rho_calculation_py(W_up, W_down, W_up_t, W_down_t, G_all, R_mNN, n1_n2_mNN, zcut_min, zcut_max, na1, na2, na3, shift_a1, shift_a2, shift_a3, tau, n_atoms, d_max, basis_tau_index, float(params["M_max_tolerance"]), tin_all, nz_t, na1_t, na2_t, kex, a1, a2, u_point_t, v_point_t, u_t, hu, hv, jac, 0.5)
    rho_time_s = time.perf_counter() - rho_start
    chi_start = time.perf_counter()
    chi = chi_calculation_py(rho, R_mNN, G_all, kex, tau, n_atoms, d_max, basis_tau_index, float(params["M_max_tolerance"]), int(params.get("Npar_Xi", 1)), e, eps0, z_point_t, b1)
    chi_time_s = time.perf_counter() - chi_start
    chi_p = np.transpose(chi, (2, 3, 0, 1, 4, 5))
    exchange_time_s = time.perf_counter() - start
    print(f"[INFO ] Exchange generator: rho/chi generation done in {exchange_time_s:.2f} s", flush=True)
    write_matlab_dataset(out_cache / "chi_WBSE.mat", "chi", chi)
    write_matlab_dataset(out_cache / "R_mNN_WBSE.mat", "R_mNN", R_mNN)
    write_matlab_dataset(out_data / "chi.mat", "chi", chi)
    write_matlab_dataset(out_data / "rho.mat", "rho", rho)
    write_matlab_dataset(out_data / "G_all.mat", "G_all", G_all)
    write_matlab_dataset(out_data / "R_mNN.mat", "R_mNN", R_mNN)
    summary = {"rho_shape": list(rho.shape), "chi_shape": list(chi.shape), "chi_p_shape": list(chi_p.shape), "R_mNN_shape": list(R_mNN.shape), "G_all_shape": list(G_all.shape), "rho_time_s": rho_time_s, "chi_time_s": chi_time_s, "exchange_time_s": exchange_time_s}
    return chi, chi_p, R_mNN, summary
