"""
read_wf_center

PURPOSE:
    Reads Wannier function center data from 'WF_centers.txt'.

DESCRIPTION:
    Python equivalent of Read_WF_center.m, returning Python/NumPy types.

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

def read_wf_center_py(wf_centers_file_path, n_atoms):
    """
    read_wf_center_py
    
    PURPOSE:
        Reads Wannier Function center coordinates and basis-to-atom mapping.
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        wf_centers_file_path (str or Path): Path to the 'WF_centers.txt' file.
        n_atoms (int): Number of atoms (obtained from structure reading).
    
    OUTPUT:
        tuple: Contains the following elements:
        wf_center (np.ndarray): N_basis x M array containing WF center data
        (typically index, atom_idx, x, y, z).
        basis_tau_index (list): A list where index `i` contains another list
        of basis function indices belonging to atom `i+1`.
        (Python equivalent of the MATLAB cell array).
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    lines = []
    try:
        with open(wf_centers_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: WF centers file not found at {wf_centers_file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading WF centers file '{wf_centers_file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if not lines:
        print(f"Error: WF centers file '{wf_centers_file_path}' seems too short.", file=sys.stderr)
        sys.exit(1)

    wf_center_list = []
    # Accept both legacy two-header files and newer files with comments or a
    # single column-name header by collecting every numeric data row.
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or stripped.startswith("#"):
            continue
        values = []
        for token in stripped.split():
            try:
                values.append(float(token.replace("D", "E").replace("d", "e")))
            except ValueError:
                break
        if values:
            wf_center_list.append(np.array(values, dtype=float))

    if not wf_center_list:
        print(f"Error: No valid WF center data found in '{wf_centers_file_path}'.", file=sys.stderr)
        sys.exit(1)

    # Convert list of arrays to a 2D NumPy array
    wf_center = np.array(wf_center_list)

    # --- Create the basis_tau_index list (equivalent to MATLAB cell array) ---
    # Initialize a list of empty lists, one for each atom
    basis_tau_index = [[] for _ in range(n_atoms)]

    # Populate the list: wf_center[:, 1] contains the atom index (1-based from file)
    # wf_center[:, 0] contains the basis function index (1-based from file)
    try:
        atom_indices = wf_center[:, 1].astype(int) # Get the second column as integer atom indices
        basis_indices = wf_center[:, 0].astype(int) # Get the first column as integer basis indices

        for basis_idx, atom_idx in zip(basis_indices, atom_indices):
            # Check if atom_idx is valid (1 to n_atoms)
            if 1 <= atom_idx <= n_atoms:
                # Python list index is 0-based, MATLAB/File index is 1-based
                basis_tau_index[atom_idx - 1].append(basis_idx)
            else:
                print(f"Warning: Found invalid atom index {atom_idx} for basis {basis_idx} in WF centers file. Skipping.", file=sys.stderr)

    except IndexError:
        print("Error: WF centers file format seems incorrect. Check columns for atom and basis indices.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while creating basis_tau_index: {e}", file=sys.stderr)
        sys.exit(1)

    # Return the NumPy array and the list of lists
    return wf_center, basis_tau_index
