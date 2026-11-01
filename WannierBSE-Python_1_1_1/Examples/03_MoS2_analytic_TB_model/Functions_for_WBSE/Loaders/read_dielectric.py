"""
read_dielectric

PURPOSE:
    Reads dielectric function data (q vs epsilon) from 'epsilon.txt'.

DESCRIPTION:
    Python equivalent of Read_dielectric.m.

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

def read_dielectric_py(epsilon_file_path):
    """
    read_dielectric_py
    
    PURPOSE:
        Reads the dielectric function data from the specified file.
    
    DESCRIPTION:
        Assumes a two-column text file format (q epsilon) without headers.
    
    INPUT ARGUMENTS:
        epsilon_file_path (str or Path): Path to the 'epsilon.txt' file.
    
    OUTPUT:
        tuple: Contains the following elements:
        q_data (np.ndarray): 1D array of q-vector magnitudes.
        eps_q_2d_data (np.ndarray): 1D array of corresponding epsilon values.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    try:
        # Use numpy.loadtxt for efficient reading of simple numeric text files
        # unpack=True reads columns into separate variables
        q_data, eps_q_2d_data = np.loadtxt(epsilon_file_path, unpack=True)

        # Basic validation
        if q_data.ndim != 1 or eps_q_2d_data.ndim != 1 or q_data.size != eps_q_2d_data.size:
            raise ValueError("Data format incorrect. Expected two columns of equal length.")

        if q_data.size == 0:
            raise ValueError("File appears empty or contains no valid numeric data.")

    except FileNotFoundError:
        print(f"Error: Dielectric file not found at {epsilon_file_path}", file=sys.stderr)
        sys.exit(1)
    except ValueError as ve:
        # loadtxt raises ValueError for parsing issues
        print(f"Error reading dielectric file '{epsilon_file_path}': {ve}", file=sys.stderr)
        print("Please ensure the file contains only two columns of numeric data.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred reading '{epsilon_file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    return q_data, eps_q_2d_data

# Example Usage (optional)
if __name__ == '__main__':
    # Assume 'epsilon.txt' is in '../parameters/' relative to this script
    script_dir = Path(__file__).parent
    epsilon_file = script_dir.parent / 'parameters' / 'epsilon.txt'
    if epsilon_file.exists():
        print(f"Testing read_dielectric_py with: {epsilon_file}")
        q_test, eps_test = read_dielectric_py(epsilon_file)
        print("Dielectric data read successfully:")
        print(f"  Number of q-points = {q_test.size}")
        print(f"  q range: [{q_test.min():.4f}, {q_test.max():.4f}]")
        print(f"  Epsilon range: [{eps_test.min():.4f}, {eps_test.max():.4f}]")
    else:
        print(f"Test dielectric file not found at {epsilon_file}")
