# =========================================================================
# WANNIERBSE: Bethe-Salpeter Equation Solver for 2D Materials
# =========================================================================
# 
# Version: 1.1.1 (Python Implementation)
# Technical Contact: https://quantum.web.nycu.edu.tw/wannierbse
# 
# DESCRIPTION:
# This script serves as the primary execution engine for the WannierBSE
# Python package. It orchestrates the ingestion of lattice geometry, 
# electronic structure data (Tight-Binding), and dielectric screening
# functions to construct and diagonalize the BSE Hamiltonian (HBSE).
#
# EXECUTION OVERVIEW:
# 1. System initialization and path configuration
# 2. Parameter ingestion and geometry parsing
# 3. Dielectric screening and k-mesh resolution
# 4. Tight-binding and electronic-state acquisition
# 5. Exchange-interaction preparation
# 6. BSE Hamiltonian construction
# 7. Eigenvalue and eigenvector solution
# 8. Exciton analysis and binding energy
# 9. Output, logging, and system cleanup
#
# REQUIREMENTS:
# - NumPy, SciPy, Joblib
# - Helper modules in 'Functions_for_WBSE/'
#
# COPYRIGHT © 2026 THE WANNIERBSE DEVELOPMENT TEAM. ALL RIGHTS RESERVED.
# =========================================================================

import os
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

# ==========================================================================
# [1] PACKAGE INITIALIZATION AND IMPORTS
# ==========================================================================
# Import the numerical libraries and internal computational kernels used by
# the top-level WannierBSE workflow.  The scientific operations themselves
# remain in Functions_for_WBSE; this file is the user-facing driver.
try:
    script_dir = Path(__file__).resolve().parent
    from Functions_for_WBSE.Loaders.read_params import read_params_py
    from Functions_for_WBSE.Loaders.read_exchange_control import read_exchange_control_py
    from Functions_for_WBSE.Loaders.read_win import read_win_py
    from Functions_for_WBSE.Loaders.read_wf_center import read_wf_center_py
    from Functions_for_WBSE.Loaders.read_dielectric import read_dielectric_py
    from Functions_for_WBSE.Dielectric.dielectric_function import dielectric_function_py
    from Functions_for_WBSE.Loaders.resolve_data_path import resolve_data_path_py
    from Functions_for_WBSE.Loaders.check_cached_kmesh_config import cached_kmesh_config_mismatch
    from Functions_for_WBSE.Loaders.validate_external_kmesh_tb_input import validate_external_kmesh_tb_input_py
    from Functions_for_WBSE.Kmesh.reciprocal_lattice_vector import reciprocal_lattice_vector_py
    from Functions_for_WBSE.Kmesh.kmesh_for_BSE import kmesh_for_BSE_py
    from Functions_for_WBSE.Kmesh.kmesh_sorter import kmesh_sorter_py, _write_sorted_kmesh_cache
    from Functions_for_WBSE.Kmesh.kmesh_band_sync import kmesh_band_sync_py
    from Functions_for_WBSE.TB.TB_for_BSE import TB_for_BSE_py, _cache_bands_exist
    from Functions_for_WBSE.Wannier_engine.wf_centers import reorder_wf_centers_py, _wf_centers_has_spinor_labels
    from Functions_for_WBSE.Wannier_engine.wf_to_mat import cleanup_wannier_xsf_py, generate_wannier_cache_py
    from Functions_for_WBSE.Loaders.read_kmesh import read_kmesh_py
    from Functions_for_WBSE.Utils.phys_constant import phys_constant_py
    from Functions_for_WBSE.Core.A_output import save_A_output_txt
    from Functions_for_WBSE.Core.generate_vc_band import generate_vc_band_py
    from Functions_for_WBSE.Loaders.read_VB import read_VB_py
    from Functions_for_WBSE.Loaders.read_CB import read_CB_py
    from Functions_for_WBSE.Core.Coulomb.singularity_and_kweight import singularity_and_kweight_py
    from Functions_for_WBSE.Utils.output_log import output_log_py
    from Functions_for_WBSE.Core.build_HBSE import build_hbse_py
    from Functions_for_WBSE.Core.Exchange.exchange_cache import load_precomputed_exchange_py
    from Functions_for_WBSE.Core.Exchange.exchange_coulomb_interaction import prepare_packed_chi_memmap
    from Functions_for_WBSE.Core.Exchange.exchange_generator import generate_exchange_tensors_py
except ImportError as e:
    print(f"Critical Error: Failed to load computational kernels. {e}", file=sys.stderr)
    sys.exit(1)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return default


def _reset_internal_kmesh_tb_cache(kmesh_cache_file: Path,
                                   tb_data_path: Path) -> None:
    """Remove only the internally generated k-mesh/TB cache group."""
    kmesh_cache_file.unlink(missing_ok=True)
    shutil.rmtree(tb_data_path, ignore_errors=True)
    tb_data_path.mkdir(parents=True, exist_ok=True)


def _display_path(path: Path, root: Path) -> str:
    """Return a stable package-relative path for console messages."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _print_header() -> str:
    """Print the standard startup header and return its timestamp."""
    separator = "=" * 59
    title = "WannierBSE v1.1.1 (Python)"
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(separator)
    print()
    print(title.center(59))
    print()
    print(f"Execution started: {started}")
    print()
    return separator


def _announce_parallel_stage(stage: str, workers: int) -> None:
    """Report a Python parallel stage without claiming a pool was created."""
    if workers > 1:
        print(f"[PRE  ] Parallel: Using {workers} workers ({stage})", flush=True)

def main():
    """Main execution pipeline for the WannierBSE solver."""
    run_start = time.perf_counter()
    console_separator = _print_header()

    # ======================================================================
    # [2] PARAMETER INGESTION AND GEOMETRY PARSING
    # ======================================================================
    # Establish the package directories, read control files, and load the
    # crystal geometry and physical constants required by later phases.
    root_path = Path(__file__).resolve().parent
    param_path = root_path / 'Parameters'
    cache_path = root_path / 'Precomputed_data'
    user_input_path = root_path / 'User_input'
    tb_data_path = cache_path / 'TB_data'
    func_path = root_path / 'Functions_for_WBSE' / 'Utils'
    out_path = root_path / 'Exciton_data'
    for directory in (param_path, cache_path, user_input_path, tb_data_path, out_path):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        params = read_params_py(param_path / 'control.txt')
        
        num_workers = int(params.get('Npar_HBSE', params.get('Number_par', 4)))
        npar_tb = int(params.get('Npar_TB', 1))
        nv = int(params['Nv'])
        nc = int(params['Nc'])
        n_val = int(params['NVAL'])
        layer_dist = float(params.get('d', 1.0))
        raw_g_dir_max = params.get('G_dir_max')
        if raw_g_dir_max is None or (isinstance(raw_g_dir_max, str) and raw_g_dir_max.strip() in {'', '[]'}):
            g_dir_max = 1.0
        else:
            try:
                g_dir_max = float(raw_g_dir_max)
            except (TypeError, ValueError) as exc:
                raise ValueError("G_dir_max must be a finite, nonnegative scalar") from exc
        exch_params = read_exchange_control_py(param_path / 'exchange_control.txt')
        exch_params['Npar_Xi'] = int(params.get('Npar_Xi', num_workers))
        npar_xi = int(exch_params['Npar_Xi'])
        exchange_interaction = _as_bool(exch_params.get('Exchange_Interaction'), False)
        use_precomputed_chi = _as_bool(exch_params.get('Use_Precomputed_Chi'), False)
        cleanup_xsf = bool(exch_params.get('cleanup_xsf', False))
        cleanup_xsf_after_success = False
        if not exchange_interaction:
            shutil.rmtree(out_path / 'Exchange_data', ignore_errors=True)

        a1, a2, a3, a_const, tau, n_atoms = read_win_py(user_input_path)
        structure_name = "wannier90.win" if (user_input_path / "wannier90.win").is_file() else "structure.txt"
        print(f"[INFO ] Structure extracted from User_input/{structure_name}")
        charge_e, eps_vac = phys_constant_py(func_path / 'physical_constants.dat')
        print(f"[INFO ] Exchange interaction: {'Enabled' if exchange_interaction else 'Disabled'}")

        # ==================================================================
        # [3] DIELECTRIC SCREENING AND K-MESH RESOLUTION
        # ==================================================================
        # Resolve user-provided or cached dielectric, k-mesh, Wannier, and
        # tight-binding data sources. Missing derived data are generated here.
        external_tb = validate_external_kmesh_tb_input_py(user_input_path, nv, nc, exchange_interaction)
        use_user_provided_kmesh_tb = external_tb.use_user_provided_kmesh_tb
        wannier_source = None

        wannier_cache_path = cache_path / 'Wannier_functions_WBSE'
        if use_user_provided_kmesh_tb:
            wf_centers_file = param_path / 'WF_centers.txt'
        else:
            wf_centers_file = wannier_cache_path / 'WF_centers_WBSE.txt'
            mandatory_wf_cache = [
                wannier_cache_path / 'WF_centers_WBSE.txt',
                wannier_cache_path / 'WF_up.mat',
                wannier_cache_path / 'WF_down.mat',
                wannier_cache_path / 'r.mat',
                wannier_cache_path / 'GridInfo.txt',
                wannier_cache_path / 'WF_AtomPosition.txt',
            ]
            centers_need_refresh = not _wf_centers_has_spinor_labels(wf_centers_file)
            cache_needs_generation = centers_need_refresh or any(not path.is_file() for path in mandatory_wf_cache)
            if exchange_interaction and not use_precomputed_chi and cache_needs_generation:
                print("[PRE  ] Wannier: Preparing real-space Wannier functions...")
                t_wannier_preprocessing = time.perf_counter()
                generate_wannier_cache_py(root_path, overwrite=True, cleanup_xsf=cleanup_xsf)
                print(
                    f"[DONE ] Wannier: Preprocessing completed in "
                    f"{time.perf_counter() - t_wannier_preprocessing:.2f} s"
                )
                if cleanup_xsf:
                    print("[WARN ] cleanup_xsf = true. The input Wannier-function XSF files in User_input/Wannier_functions_xsf will be deleted after the calculation completes successfully.")
                    cleanup_xsf_after_success = True
                wannier_source = "preprocessed during current run from XSF files"
            elif exchange_interaction and not use_precomputed_chi:
                print("[CACHE] Wannier: Using preprocessed data")
                wannier_source = "loaded from precomputed Wannier-function cache"
            elif centers_need_refresh:
                print("[GEN  ] Wannier: Generating WF_centers_WBSE.txt...")
                wannier_cache_path.mkdir(parents=True, exist_ok=True)
                reorder_wf_centers_py(param_path / 'WF_centers.txt', wf_centers_file)
            elif exchange_interaction and use_precomputed_chi:
                print("[CACHE] Wannier: Using preprocessed data")
                wannier_source = "not used; precomputed exchange tensors loaded"
            elif not exchange_interaction:
                pass

        wf_center, basis_map = read_wf_center_py(wf_centers_file, int(n_atoms))
        nb_basis = wf_center.shape[0]
        kmesh_cache_file = cache_path / 'kmesh_WBSE.txt'
        temp_topology_path = tb_data_path / 'temp'

        eps_file, eps_tag = resolve_data_path_py('epsilon.txt', user_input_path, cache_path)
        if eps_tag == 'GEN  ' or eps_file is None:
            dielectric_start = time.perf_counter()
            print("[GEN  ] Dielectric: Generating screening function...")
            eps_file = cache_path / "epsilon_WBSE.txt"
            dielectric_function_py(
                param_path / "dielectric_control.txt",
                func_path / "physical_constants.dat",
                eps_file,
            )
            dielectric_source = "computed during current run"
            print(f"[DONE ] Dielectric: Generated in {time.perf_counter() - dielectric_start:.2f} s")
        else:
            source_label = "LOAD " if eps_tag == 'LOAD ' else "CACHE"
            print(f"[{source_label}] Dielectric: Using {_display_path(eps_file, root_path)}")
            dielectric_source = "provided by user" if eps_tag == 'LOAD ' else "loaded from precomputed data"
        q_data, eps_q2d = read_dielectric_py(eps_file)

        cache_bands_exist = _cache_bands_exist(tb_data_path, nv, nc)
        kmesh_file, kmesh_tag = resolve_data_path_py('kmesh.txt', user_input_path, cache_path)
        kmesh_cache_stale = kmesh_tag == 'CACHE' and cached_kmesh_config_mismatch(
            kmesh_file, param_path / 'kmesh_control.txt')
        stale_kmesh_regeneration = False
        if kmesh_cache_stale and not external_tb.has_user_kmesh:
            print('[GEN  ] Momentum grid: Cached configuration changed; regenerating mesh and TB data...')
            _reset_internal_kmesh_tb_cache(kmesh_cache_file, tb_data_path)
            cache_bands_exist = False
            kmesh_file = None
            kmesh_tag = 'GEN  '
            stale_kmesh_regeneration = True
        is_external_kmesh = False
        force_tb_regeneration = external_tb.has_user_kmesh and not use_user_provided_kmesh_tb

        if use_user_provided_kmesh_tb:
            print("[LOAD ] Momentum grid: Using User_input/kmesh.txt")
            k_fixed, index_map, header = kmesh_sorter_py(external_tb.user_kmesh_path)
            _write_sorted_kmesh_cache(kmesh_cache_file, header, k_fixed)
            kmesh_band_sync_py(external_tb.user_bands_path, tb_data_path, index_map, nv, nc, nb_basis, k_fixed)
            kmesh_file = kmesh_cache_file
            is_external_kmesh = True
            band_source = tb_data_path
            kmesh_source = "provided by user"
            tb_source = "provided by user"
        else:
            if external_tb.has_user_kmesh:
                print("[LOAD ] Momentum grid: Using User_input/kmesh.txt")
                k_fixed, index_map, header = kmesh_sorter_py(external_tb.user_kmesh_path)
                _write_sorted_kmesh_cache(kmesh_cache_file, header, k_fixed)
                kmesh_file = kmesh_cache_file
                is_external_kmesh = True
                kmesh_source = "provided by user"
            elif kmesh_tag == 'GEN  ' or kmesh_file is None:
                print("[GEN  ] Momentum grid: Generating sampling mesh...")
                try:
                    kmesh_file = kmesh_for_BSE_py(root_path)
                except Exception:
                    if stale_kmesh_regeneration:
                        _reset_internal_kmesh_tb_cache(kmesh_cache_file, tb_data_path)
                    raise
                kmesh_source = "computed during current run"
            else:
                print("[CACHE] Momentum grid: Using Precomputed_data/kmesh_WBSE.txt")
                kmesh_source = "loaded from precomputed data"

            if cache_bands_exist and not force_tb_regeneration:
                print("[CACHE] Bands: Using Precomputed_data/TB_data/")
                band_source = tb_data_path
                tb_source = "loaded from precomputed data"
            else:
                if force_tb_regeneration:
                    print("[GEN  ] User TB_data incomplete or absent. Regenerating TB on the supplied k-mesh...")
                print("[GEN  ] Bands: Generating from Wannier90...")
                tb_context = {
                    "kmeshFile": kmesh_file,
                    "wannierWinFile": user_input_path / "wannier90.win",
                    "structureFile": user_input_path / "structure.txt",
                    "wannierHrFile": user_input_path / "wannier90_hr.dat",
                    "wfCentersFile": param_path / "WF_centers.txt",
                    "controlFile": param_path / "control.txt",
                    "wtbControlFile": param_path / "WTB_control.txt",
                    "tbOutputDir": tb_data_path,
                    "exchangeInteraction": exchange_interaction,
                }
                tb_start = time.perf_counter()
                _announce_parallel_stage("TB", npar_tb)
                try:
                    TB_for_BSE_py(tb_context)
                    if stale_kmesh_regeneration and not _cache_bands_exist(tb_data_path, nv, nc):
                        raise RuntimeError(
                            f"Regenerated TB cache is incomplete for Nv={nv} and Nc={nc}."
                        )
                except Exception:
                    if stale_kmesh_regeneration:
                        _reset_internal_kmesh_tb_cache(kmesh_cache_file, tb_data_path)
                    raise
                print(f"[DONE ] Bands: Generated in {time.perf_counter() - tb_start:.2f} s", flush=True)
                band_source = tb_data_path
                tb_source = "computed during current run"

        f_scale, nv_coarse, nv_dense, n_co, n_de, k_grid = read_kmesh_py(kmesh_file)
        n_kpt = k_grid.shape[0]
        b1, b2 = reciprocal_lattice_vector_py(a1, a2)

        # ==================================================================
        # [4] TIGHT-BINDING AND ELECTRONIC-STATE ACQUISITION
        # ==================================================================
        # Load valence/conduction energies and Bloch-state coefficients on the
        # resolved k mesh.
        ev_energy, cv_coeffs = read_VB_py(band_source, nv, n_kpt, charge_e, nb_basis)
        ec_energy, cc_coeffs = read_CB_py(band_source, nc, n_kpt, charge_e, nb_basis)

        # TB topology sidecars are needed only while generating/reading the
        # TB bands; remove them once the coefficients have been ingested.
        if temp_topology_path.is_dir():
            shutil.rmtree(temp_topology_path)

        # ==================================================================
        # [5] EXCHANGE-INTERACTION PREPARATION
        # ==================================================================
        # Load or generate the short-range exchange tensors. This block is
        # skipped when Exchange_Interaction is disabled.
        chi_p = None
        r_mnn = None
        packed_chi_descriptor = None
        exchange_summary = {}
        if exchange_interaction:
            if use_precomputed_chi:
                print("[CACHE] Exchange: Loading precomputed chi/R_mNN tensors...")
                _, chi_p, r_mnn = load_precomputed_exchange_py(cache_path / 'Exchange_interaction', expected_nb=nb_basis)
                print(f"[DONE ] Exchange: Loaded chi_p {chi_p.shape} and R_mNN {r_mnn.shape}")
            else:
                print("[GEN  ] Exchange: Calculating chi/R_mNN tensors...")
                _announce_parallel_stage("Exchange", npar_xi)
                _, chi_p, r_mnn, exchange_summary = generate_exchange_tensors_py(
                    root_path, a1, a2, tau, int(n_atoms), basis_map, b1, b2,
                    float(charge_e), float(eps_vac), exch_params
                )

            # χ is already available at this point.  Pack its (r1, r2)
            # slices once as read-only S1 matrices; workers receive only the
            # descriptor and reuse their local maps throughout Vx assembly.
            packed_file = out_path / f"vx_chi_packed_{os.getpid()}_{time.time_ns()}.dat"
            packed_chi_descriptor = prepare_packed_chi_memmap(chi_p, packed_file)
            print(
                f"[INFO ] Exchange: Packed chi {packed_chi_descriptor['shape']} "
                f"to read-only backing file {packed_chi_descriptor['path']}"
            )

        # ==================================================================
        # [6] BSE HAMILTONIAN CONSTRUCTION
        # ==================================================================
        # Construct the raw interaction matrix and assemble the weighted
        # Hamiltonian H_BSE = K + w_ij*(Vd + Vx).
        print("[SOLVE] Constructing HBSE matrix...")
        _announce_parallel_stage("HBSE", num_workers)
        wq0, k_weights = singularity_and_kweight_py(n_de, n_co, nv_coarse, nv_dense, layer_dist, q_data, eps_q2d, a_const, f_scale)
        vc_pairs = generate_vc_band_py(nv, nc)

        start_time_hbse = time.perf_counter()
        try:
            hbse_interaction = build_hbse_py(
                vc_pairs, nv, nc, n_kpt, num_workers,
                ec_energy, ev_energy, k_grid, b1, b2, layer_dist, q_data, eps_q2d,
                wq0, k_weights, tau, basis_map, wf_center, int(n_atoms),
                cc_coeffs, cv_coeffs, float(charge_e), float(eps_vac),
                exchange_interaction=exchange_interaction, chi_p=chi_p, R_mNN=r_mnn,
                G_dir_max=g_dir_max, chi_packed_descriptor=packed_chi_descriptor,
            )
        finally:
            if packed_chi_descriptor is not None:
                try:
                    Path(packed_chi_descriptor["path"]).unlink(missing_ok=True)
                except OSError as cleanup_error:
                    print(f"[WARN ] Could not remove temporary packed chi file: {cleanup_error}")

        # BSE assembly: build_hbse_py above has already formed the raw
        # interaction matrix V = Vd + Vx (Vx is included only when exchange
        # is enabled).  The final Hamiltonian is assembled in two steps:
        #
        #   1) H_BSE <- w_ij * (Vd + Vx)
        #   2) H_BSE <- H_BSE + K,     K = Ec(k) - Ev(k)
        #
        # Therefore, elementwise, the result is
        #
        #   H_BSE,ij = K_ij + sqrt(w_i*w_j)/(2*pi)^2 * (Vd,ij + Vx,ij).
        #
        # Apply symmetric k-mesh quadrature to every sector block.  The raw
        # interaction matrix is block-ordered with one contiguous k block per
        # (v,c) sector, so the Kronecker factor repeats the same k-by-k
        # symmetric weight matrix for every sector-block pair.
        weight_matrix_k = np.sqrt(k_weights[:, None] * k_weights[None, :]) / (4 * np.pi**2)
        n_sectors = nc * nv
        symmetric_weight_matrix = np.kron(
            np.ones((n_sectors, n_sectors), dtype=weight_matrix_k.dtype),
            weight_matrix_k,
        )
        # Step 1: apply w_ij = sqrt(w_i*w_j)/(2*pi)^2 to (Vd + Vx).
        hbse_full = hbse_interaction * symmetric_weight_matrix

        for idx, (v, c) in enumerate(vc_pairs):
            offset = idx * n_kpt
            # Step 2: add the independent-particle term K = Ec - Ev.
            hbse_full[offset:offset+n_kpt, offset:offset+n_kpt] += np.diag(ec_energy[:, c-1] - ev_energy[:, v-1])

        time_hbse = time.perf_counter() - start_time_hbse
        print(f"[DONE ] BSE matrix constructed in {time_hbse:.2f} s")

        # ==================================================================
        # [7] EIGENVALUE AND EIGENVECTOR SOLUTION
        # ==================================================================
        print("[SOLVE] Diagonalizing HBSE matrix...")
        start_time_eigs = time.perf_counter()
        
        matrix_size = hbse_full.shape[0]
        hbse_operator = LinearOperator(
            shape=hbse_full.shape,
            dtype=hbse_full.dtype,
            matvec=lambda vec: hbse_full @ vec
        )
        n_val_safe = min(n_val, matrix_size - 2)
        subspace_dim = min(matrix_size, max(2 * n_val_safe + 20, n_val_safe + 2))
        rng = np.random.default_rng()

        eig_vals, eig_vecs = eigs(
            hbse_operator, k=n_val_safe, which='SR', tol=1e-14, ncv=subspace_dim,
            v0=rng.random(matrix_size) + 1j * rng.random(matrix_size)
        )

        # Sorting and post-processing
        sort_idx = np.argsort(np.real(eig_vals))
        ex_energies = np.real(eig_vals[sort_idx])
        a_vecs = eig_vecs[:, sort_idx]

        time_eigs = time.perf_counter() - start_time_eigs
        print(f"[DONE ] Diagonalization completed in {time_eigs:.2f} s")

        # ==================================================================
        # [8] EXCITON ANALYSIS AND BINDING ENERGY
        # ==================================================================
        # Sort the exciton energies/eigenvectors and convert the eigenvectors
        # to the output normalization used by the post-processing files.
        # Scaling eigenvectors for normalization consistency
        norm_scale = np.tile(np.sqrt((4 * np.pi**2) / k_weights), nc * nv).reshape(-1, 1)
        a_final = a_vecs * norm_scale

        # ==================================================================
        # [9] OUTPUT, LOGGING, AND SYSTEM CLEANUP
        # ==================================================================
        # Export text outputs to Exciton_data.
        np.savetxt(out_path / 'Ex.txt', ex_energies, fmt='%.18e', header="Exciton Energies (eV)")
        np.savetxt(out_path / 'wi.txt', k_weights, fmt='%.18e', header="K-point Weights")
        
        a_reshaped = np.empty((a_final.shape[0], a_final.shape[1] * 2))
        a_reshaped[:, 0::2], a_reshaped[:, 1::2] = a_final.real, a_final.imag
        np.savetxt(out_path / 'A.txt', a_reshaped, fmt='%.18e', header="Scaled Eigenvectors (Real, Imag)")

        k_save = np.hstack([k_grid, np.zeros((n_kpt, 1))]) if k_grid.shape[1] == 2 else k_grid
        np.savetxt(out_path / 'k.txt', k_save, fmt='%.18e', header="K-points (1/Angstrom)")

        if is_external_kmesh and use_user_provided_kmesh_tb:
            archive_bands = out_path / 'TB_Bands'
            archive_bands.mkdir(parents=True, exist_ok=True)
            for pattern in ('v*_TB_WBSE.txt', 'c*_TB_WBSE.txt'):
                for band_file in tb_data_path.glob(pattern):
                    shutil.move(str(band_file), str(archive_bands / band_file.name))
            for band_file in archive_bands.glob('*_WBSE.txt'):
                renamed_file = band_file.with_name(band_file.name.replace('_WBSE', ''))
                band_file.rename(renamed_file)
            kmesh_cache_file.unlink(missing_ok=True)

        output_log_py(
            out_path,
            n_kpt,
            nv,
            nc,
            num_workers,
            np.array([time_hbse, time_eigs, time_hbse + time_eigs]),
            exchange_params=exch_params,
            parallel_params={
                "Npar_TB": int(params.get("Npar_TB", 1)),
                "Npar_Xi": int(params.get("Npar_Xi", num_workers)),
                "Npar_HBSE": int(params.get("Npar_HBSE", num_workers)),
            },
            exchange_times=exchange_summary,
            total_execution_time=time.perf_counter() - run_start,
            provenance={
                "kmesh_source": kmesh_source,
                "tb_source": tb_source,
                "dielectric_source": dielectric_source,
                "wannier_source": wannier_source,
            },
        )
        if cleanup_xsf_after_success:
            cleanup_wannier_xsf_py(root_path)
        total_execution_time = time.perf_counter() - run_start
        print(f"[DONE ] BSE stage completed in {time_hbse + time_eigs:.2f} s")
        print(f"[DONE ] Total execution time: {total_execution_time:.2f} s")
        print("[DONE ] Execution successful. Results: Exciton_data/")
        print()
        print(console_separator)

    except Exception as e:
        print(f"\n[ERROR] Calculation aborted: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
