"""
read_CB

PURPOSE:
    Reads all conduction band (CB) tight-binding data from files.

DESCRIPTION:
    Python equivalent of READ_CB.m, using the new read_TBdata_py parser.

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
from pathlib import Path
import numpy as np

# Import the core parser from the same directory
try:
    # Use relative import for package structure
    from .read_TBdata import read_TBdata_py
except ImportError:
    # Fallback for direct execution/testing (e.g., if __name__ == '__main__')
    try:
        from read_TBdata import read_TBdata_py
    except ImportError:
        print("Error: Cannot import read_TBdata_py. Ensure read_TBdata.py is in the same directory.", file=sys.stderr)
        sys.exit(1)

def read_CB_py(tb_data_path, Nc, N_kpt, e, NB, verbose=False):
    """
    read_CB_py
    
    PURPOSE:
        Loads and parses all conduction band (CB) files (c1_TB_WBSE.txt, ...).
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        tb_data_path (Path or str): Path object or string to the 'Precomputed_data/TB_data' directory.
        Nc (int): Number of conduction bands to read.
        N_kpt (int): Number of k-points.
        e (float): Elementary charge (for scaling energy).
        NB (int): Number of basis functions (orbitals).
        verbose (bool, optional): If True, print info logs. Defaults to False.
    
    OUTPUT:
        tuple: Contains the following elements:
        Ec (np.ndarray): 2D array of CB energies, shape (N_kpt, Nc).
        Cc (np.ndarray): 3D array of CB coefficients, shape (N_kpt, NB, Nc).
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    # Ensure tb_data_path is a Path object for easier file joining
    if not isinstance(tb_data_path, Path):
        tb_data_path = Path(tb_data_path)

    # Initialize variables for linter (in case loops don't run)
    Ec = None
    Cc = None
    i = 0 # Initialize 'i' to fix "possibly unbound" linter warning

    try:
        # Pre-allocate NumPy arrays
        Ec = np.zeros((N_kpt, Nc))
        Cc = np.zeros((N_kpt, NB, Nc), dtype=complex)

        if verbose:
            print(f"Reading {Nc} conduction band(s)...")
        for i in range(1, Nc + 1): # Loop from 1 to Nc
            cb_file_name = f"c{i}_TB_WBSE.txt"
            cb_file_path = tb_data_path / cb_file_name

            # Read all lines from the file
            try:
                with open(cb_file_path, 'r') as f:
                    raw_lines = f.readlines()
            except FileNotFoundError:
                print(f"Error: Conduction band file not found at {cb_file_path}", file=sys.stderr)
                sys.exit(1)
            except Exception as err: # Rename 'e' to 'err'
                print(f"Error reading {cb_file_name}: {err}", file=sys.stderr)
                sys.exit(1)

            # Parse the loaded data using our new Python function
            En, Cn = read_TBdata_py(N_kpt, raw_lines, e, NB, verbose=verbose)

            # Store results in the main arrays
            # # (i-1) for 0-based Python indexing
            Ec[:, i-1] = En
            Cc[:, :, i-1] = Cn

            if i > 0: # Check if i > 0, meaning the loop started
                if verbose:
                    print(f"  (Processing {cb_file_name})")
        
    except Exception as err: # Rename 'e' to 'err'
        print(f"Error processing conduction band files: {err}", file=sys.stderr)
        sys.exit(1)

    return Ec, Cc
