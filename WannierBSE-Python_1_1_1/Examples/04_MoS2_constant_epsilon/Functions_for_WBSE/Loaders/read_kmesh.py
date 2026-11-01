"""
read_kmesh

PURPOSE:
    Reads k-mesh parameters and 2D k-points from 'kmesh.txt'.

DESCRIPTION:
    Python equivalent of Read_kmesh.m.

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

def read_kmesh_py(kmesh_file_path):
    """
    read_kmesh_py
    
    PURPOSE:
        Reads k-mesh parameters and 2D k-points (kx, ky) from the specified file.
    
    DESCRIPTION:
        Line 1: f (float)
        Line 2: Nv_coarse (int)
        Line 3: Nv_dense (int)
        Line 4: N_co (int)
        Line 5: N_de (int)
        Following lines: kx ky kz (3 columns per line)
    
    INPUT ARGUMENTS:
        kmesh_file_path (str or Path): Path to the 'kmesh.txt' file.
    
    OUTPUT:
        tuple: Contains the following elements:
        f (float): Fraction of dense region.
        Nv_coarse (int): Number of k-points along coarse grid axis.
        Nv_dense (int): Number of k-points along dense grid axis.
        N_co (int): Number of coarse k-points.
        N_de (int): Number of dense k-points.
        k (np.ndarray): Nx2 array of k-points (kx, ky).
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    try:
        with open(kmesh_file_path, 'r') as fid:
            # Read parameters first
            # Line 1: dense_range (int/float)
            # Line 2: Nv_coarse (int)
            # Line 3: Nv_dense (int)
            # Line 4: N_co (int)
            # Line 5: N_de (int)
            dense_range = float(fid.readline().strip())
            Nv_coarse = int(fid.readline().strip())
            Nv_dense = int(fid.readline().strip())
            N_co = int(fid.readline().strip())
            N_de = int(fid.readline().strip())

            # Calculate f (scaling fraction) matching MATLAB Read_kmesh.m
            # f = Nv_coarse / dense_range
            f = float(Nv_coarse) / dense_range

            # Read remaining lines as k-points (kx, ky, kz)
            # Use numpy.loadtxt on the rest of the file handle
            # Use usecols=(0, 1) to read only the first two columns (kx, ky)
            k = np.loadtxt(fid, usecols=(0, 1))

        # Validate k-points array
        if k.ndim != 2 or k.shape[1] != 2:
            raise ValueError(f"Expected Nx2 array for k-points, but got shape {k.shape}")

        # Optional: Validate counts (N_co + N_de should equal total k-points)
        if k.shape[0] != (N_co + N_de):
            print(f"Warning: Number of k-points read ({k.shape[0]}) does not match N_co ({N_co}) + N_de ({N_de}). Check kmesh.txt.", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: K-mesh file not found at {kmesh_file_path}", file=sys.stderr)
        sys.exit(1)
    except ValueError as ve:
        # Catch errors from float(), int(), or loadtxt parsing
        print(f"Error reading or parsing k-mesh file '{kmesh_file_path}': {ve}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred reading '{kmesh_file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    return f, Nv_coarse, Nv_dense, N_co, N_de, k

# Example Usage (optional)
if __name__ == '__main__':
    # Assume 'kmesh.txt' is in '../TB_data/' relative to this script
    script_dir = Path(__file__).parent
    kmesh_file = script_dir.parent / 'TB_data' / 'kmesh.txt'
    if kmesh_file.exists():
        print(f"Testing read_kmesh_py with: {kmesh_file}")
        f_test, Nvc_test, Nvd_test, Nc_test, Nd_test, k_test = read_kmesh_py(kmesh_file)
        print("K-mesh data read successfully:")
        print(f"  f = {f_test}")
        print(f"  Nv_coarse = {Nvc_test}")
        print(f"  Nv_dense = {Nvd_test}")
        print(f"  N_co = {Nc_test}")
        print(f"  N_de = {Nd_test}")
        print(f"  k shape = {k_test.shape}")
        print(f"  k (first 5 points) =\n{k_test[:5, :]}")
    else:
        print(f"Test k-mesh file not found at {kmesh_file}")
