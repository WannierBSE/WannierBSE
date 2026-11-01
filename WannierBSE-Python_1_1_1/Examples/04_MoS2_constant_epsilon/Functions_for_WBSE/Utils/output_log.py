"""
output_log

PURPOSE:
    Writes a log file summarizing the BSE calculation.

DESCRIPTION:
    Python equivalent of output_log.m.

INPUT ARGUMENTS:
    See the annotated signature and detailed description.

OUTPUT:
    See the annotated signature and detailed description.

PHYSICS/NUMERICAL NOTES:
    Preserves the numerical conventions, units, indexing, and array ordering established by the caller.

SIDE EFFECTS:
    None unless explicitly documented by this function.
"""
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

def output_log_py(
    DataPath,
    N_kpt,
    Nv,
    Nc,
    Npar,
    Time_vec,
    *,
    exchange_params=None,
    parallel_params=None,
    exchange_times=None,
    total_execution_time=None,
    provenance=None,
):
    """
    output_log_py
    
    PURPOSE:
        Writes a log file with computation details.
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        DataPath (Path or str): Path to the output 'Exciton_data' directory.
        N_kpt (int): Number of k-points.
        Nv (int): Number of valence bands.
        Nc (int): Number of conduction bands.
        Npar (int): Number of parallel workers (n_jobs).
        Time_vec (np.ndarray): 1D array of timing data.
        Expected format for pure Python (3 elements):
        [Build_Python_joblib, Diag_SciPy, Total]
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    log_file_path = Path(DataPath) / 'WBSE_log.txt'

    try:
        with open(log_file_path, 'w') as f:
            t_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Detailed report format matching the MATLAB WBSE_log.txt.
            if exchange_params is not None:
                exchange_params = dict(exchange_params)
                parallel_params = dict(parallel_params or {})
                exchange_times = dict(exchange_times or {})
                provenance = dict(provenance or {})
                interaction = bool(exchange_params.get("Exchange_Interaction", False))
                source = "precomputed from cache" if bool(exchange_params.get("Use_Precomputed_Chi", False)) else "computed during current run"
                solver_time = float(Time_vec[0]) + float(Time_vec[1])
                f.write("WannierBSE calculation report\n")
                f.write(f"Generated at: {t_now}\n\n")
                f.write("==================== INPUT DATA PROVENANCE ====================\n")
                f.write(f"k-mesh source                                  = {provenance.get('kmesh_source', '')}\n")
                f.write(f"TB data source                                 = {provenance.get('tb_source', '')}\n")
                f.write(f"dielectric screening source                    = {provenance.get('dielectric_source', '')}\n")
                if interaction:
                    f.write(f"Wannier-function source                        = {provenance.get('wannier_source', '')}\n")
                f.write("\n")
                f.write("==================== BSE BASIS ====================\n")
                f.write(f"Number of k-points                            = Nk = {N_kpt}\n")
                f.write(f"Valence bands used                            = Nv = {Nv}\n")
                f.write(f"Conduction bands used                         = Nc = {Nc}\n\n")
                f.write("==================== COULOMB INTERACTION ====================\n")
                f.write(f"Short-range exchange interaction              = {'enabled' if interaction else 'disabled'}\n")
                if interaction:
                    f.write(f"Exchange tensor source                        = {source}\n\n")
                    f.write(f"Nearest-neighbor Wannier shell                M_max                  = {exchange_params.get('M_max', '')}\n")
                    f.write(f"Largest short-range G-vector cutoff           G_max                  = {exchange_params.get('G_max', '')}\n")
                    f.write(f"Wannier truncation probability cutoff         tol_probability        = {exchange_params.get('tol_probability', '')}\n")
                    f.write(f"M-neighbor distance tolerance                 M_max_tolerance        = {exchange_params.get('M_max_tolerance', '')}\n")
                    f.write(f"Nearest-neighbor distance tolerance           nn_distance_tolerance  = {exchange_params.get('nn_distance_tolerance', '')}\n\n")
                else:
                    f.write("\n")
                f.write("==================== CPU USAGE ====================\n")
                f.write(f"TB calculations                               Npar_TB                = {parallel_params.get('Npar_TB', '')}\n")
                if interaction and not bool(exchange_params.get("Use_Precomputed_Chi", False)):
                    f.write(f"Exchange tensor calculation                   Npar_Xi                = {parallel_params.get('Npar_Xi', '')}\n")
                f.write(f"BSE Hamiltonian construction                  Npar_HBSE              = {parallel_params.get('Npar_HBSE', '')}\n\n")
                f.write("==================== COMPUTATION TIME ====================\n")
                if interaction and not bool(exchange_params.get("Use_Precomputed_Chi", False)):
                    f.write(f"rho overlap-density calculation               = {float(exchange_times.get('rho_time_s', 0.0)):12.3f} sec\n")
                    f.write(f"short-range exchange tensor calculation       = {float(exchange_times.get('chi_time_s', 0.0)):12.3f} sec\n")
                f.write(f"BSE Hamiltonian matrix construction           = {float(Time_vec[0]):12.3f} sec\n")
                f.write(f"BSE Hamiltonian diagonalization               = {float(Time_vec[1]):12.3f} sec\n")
                f.write(f"Total BSE solver stage                        = {solver_time:12.3f} sec\n")
                f.write(f"Total WBSE execution                          = {float(total_execution_time if total_execution_time is not None else solver_time):12.3f} sec\n")
                return

            f.write(f"------------------ k-points and bands and included in BSE ------------------ \n")
            f.write(f"number of k-points Nk = {N_kpt}\n")
            f.write(f"number of valence and conduction bands (Nv,Nc) = ( {Nv} , {Nc} )\n\n\n")

            f.write(f"------------------ CPU usage ------------------ \n")
            f.write(f"The number of cores used (n_jobs) = {Npar}\n\n")

            f.write(f"------------------ Computation time (Full Python) ------------------ \n")

            # --- FIX: Correctly handle 3-element or 4-element Time_vec ---
            
            if len(Time_vec) == 3:
                f.write(f"Computation time of constructing BSE matrix (Python/joblib) = {Time_vec[0]:12.3f} sec.\n")
                f.write(f"Computation time of diagonalizing BSE matrix (SciPy)         = {Time_vec[1]:12.3f} sec.\n")
                f.write(f"                                                    Total computation time = {Time_vec[2]:12.3f} sec.\n")
            elif len(Time_vec) == 4: # Handle old hybrid log (just in case)
                f.write(f"Computation time of constructing BSE matrix (MATLAB) = {Time_vec[0]:12.3f} sec.\n")
                f.write(f"Computation time of converting matrix (Python)       = {Time_vec[1]:12.3f} sec.\n")
                f.write(f"Computation time of diagonalizing BSE matrix (SciPy)  = {Time_vec[2]:12.3f} sec.\n")
                f.write(f"                                                    Total computation time = {Time_vec[3]:12.3f} sec.\n")
            else:
                f.write(f"Computation time (unexpected vector format): {Time_vec}\n")
            # --- END FIX ---

    except Exception as e:
        print(f"Error writing log file '{log_file_path}': {e}", file=sys.stderr)

# Example Usage (optional)
if __name__ == '__main__':
    # Test the function
    _test_path = Path.cwd() / 'Exciton_data_TEST'
    _test_path.mkdir(exist_ok=True)
    _N_kpt = 1000
    _Nv = 2
    _Nc = 2
    _Npar = 8
    _Time_vec_test = np.array([120.5, 10.1, 130.6])

    print(f"Testing output_log_py, writing to {_test_path} / 'BSE_log.txt'...")
    output_log_py(_test_path, _N_kpt, _Nv, _Nc, _Npar, _Time_vec_test)
    print("Test log written.")
