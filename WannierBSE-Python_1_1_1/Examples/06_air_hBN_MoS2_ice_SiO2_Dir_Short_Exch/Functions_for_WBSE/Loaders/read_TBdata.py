"""
read_TBdata

PURPOSE:
    Parses the raw data from a single TB data file (e.g., V1_BSE.txt).

DESCRIPTION:
    Python equivalent of read_TBdata.m, but vectorized for efficiency.
    Handles files with 2 or 3 columns for coefficients.

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

def read_TBdata_py(N_kpt, raw_lines, e, NB, verbose=False):
    """
    read_TBdata_py
    
    PURPOSE:
        Parses a list of raw text lines from a tight-binding data file
    
    DESCRIPTION:
        into energy (En) and coefficient (Cn) arrays.
    
        This function is a vectorized replacement for the loop in
        Read_TBdata.m.
    
    INPUT ARGUMENTS:
        N_kpt (int): Number of k-points.
        raw_lines (list): List of strings (lines) from the .txt file.
        e (float): Elementary charge (for scaling energy).
        NB (int): Number of basis functions (orbitals).
        verbose (bool, optional): If True, print info logs. Defaults to
        False.
    
    OUTPUT:
        tuple: Contains the following elements:
        En (np.ndarray): 1D array of energies, shape (N_kpt,).
        Cn (np.ndarray): 2D array of complex coefficients, shape
        (N_kpt, NB).
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    # Each k-point block has 1 row (k-point) + 1 row (energy) + NB rows (coeffs)
    block_size = NB + 2
    num_blocks = len(raw_lines) // block_size

    # Initialize variables for the except block
    all_coeffs_list = []

    if num_blocks != N_kpt:
        print(f"Warning: k-point mismatch in read_TBdata_py. Expected {N_kpt}, found {num_blocks}.", file=sys.stderr)
        # N_kpt = num_blocks # Trust the file's block count

    if N_kpt == 0:
        print(f"Error: No data blocks found in file.", file=sys.stderr)
        sys.exit(1)

    try:
        # Pre-allocate arrays
        En_list = []
        all_coeffs_list = [] # Will store flat list of coefficients

        line_idx = 0
        for ik in range(num_blocks):
            # Skip k-point line (index 0)
            line_idx += 1

            # Read energy line (index 1)
            En_list.append(float(raw_lines[line_idx].split()[0]))
            line_idx += 1

            # Read NB coefficient lines (indices 2 to NB+1)
            coeff_lines = raw_lines[line_idx : line_idx + NB]
            all_coeffs_list.extend(coeff_lines)
            line_idx += NB

        # Convert energy list to numpy array and scale
        En = np.array(En_list) * e

        # Convert all coefficient lines to a numpy array at once
        # Use a more robust split() method which handles all whitespace
        coeffs_flat_str = " ".join(all_coeffs_list)
        coeffs_data = np.array(coeffs_flat_str.split(), dtype=float)

        # Figure out how many columns were read (e.g., 2 or 3)
        # Avoid division by zero
        denom = num_blocks * NB
        if denom == 0:
            raise ValueError(f"Internal error: num_blocks * NB is zero. (num_blocks={num_blocks}, NB={NB})")
            
        num_cols = len(coeffs_data) // denom

        if num_cols < 2:
            # Check if maybe the file format is different than expected
            actual_per_block = len(coeffs_data) // num_blocks if num_blocks > 0 else 0
            raise ValueError(f"Coefficient data is malformed. Expected at least {NB*2} values per k-point block (NB={NB} * 2 columns), but found only {actual_per_block}. Check if NB is correct.")

        # Reshape to (num_blocks, NB, num_cols) and take only the first 2
        try:
            coeffs_data = coeffs_data.reshape(num_blocks, NB, num_cols)[:, :, :2]
        except ValueError as e:
            raise ValueError(f"Reshape failed: {e}. (num_blocks={num_blocks}, NB={NB}, num_cols={num_cols}, total_size={len(coeffs_data)})")

        # Separate real and imaginary parts
        Cn_real = coeffs_data[:, :, 0]
        Cn_imag = coeffs_data[:, :, 1]

        # Create complex array
        Cn = Cn_real + 1j * Cn_imag
        
        # Verify shape
        if Cn.shape != (num_blocks, NB):
             Cn = Cn.reshape(num_blocks, NB)
        
        # If num_blocks != N_kpt, we might need to pad/truncate, but 
        # normally they match.
        if num_blocks != N_kpt:
            # Simple handling: truncate or pad with zeros
            if num_blocks > N_kpt:
                return En[:N_kpt], Cn[:N_kpt, :]
            else:
                En_final = np.zeros(N_kpt)
                Cn_final = np.zeros((N_kpt, NB), dtype=complex)
                En_final[:num_blocks] = En
                Cn_final[:num_blocks, :] = Cn
                return En_final, Cn_final

        return En, Cn

    except ValueError as ve:
        print(f"Error: Failed to parse block structure in read_TBdata_py.", file=sys.stderr)
        print(f"  Details: {ve}", file=sys.stderr)
        print(f"  N_kpt: {N_kpt}, NB: {NB}, BlockSize: {block_size}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error in read_TBdata_py: {e}", file=sys.stderr)
        sys.exit(1)
