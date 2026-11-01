"""
A_output

PURPOSE:
    Calculates scaled exciton eigenvectors (A) and saves them to a text file.

DESCRIPTION:
    Modified from original A_output.py to write .txt instead of returning array.

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

def save_A_output_txt(output_file_path, A, Nc, Nv, wi, N_kpt):
    """
    save_A_output_txt
    
    PURPOSE:
        Scales the eigenvectors (A) by the k-point weights (wi) and saves
    
    DESCRIPTION:
        the result to a text file.
    
        Each complex number is saved as two columns: real part, imaginary part.
    
    INPUT ARGUMENTS:
        output_file_path (str or Path): Path to the output text file (e.g., 'A.txt').
        A (np.ndarray): Raw eigenvector matrix from diagonalization,
        shape (Nc*Nv*N_kpt, NVAL).
        Nc (int): Number of conduction bands.
        Nv (int): Number of valence bands.
        wi (np.ndarray): K-point weights, shape (N_kpt,).
        N_kpt (int): Number of k-points.
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    try:
        # 1. Ensure wi is a 1D array
        if wi.ndim > 1:
            wi = wi.flatten()

        # 2. Calculate the scale vector (shape (N_kpt,))
        # scale = sqrt((4*pi^2)./wi)
        scale = np.sqrt((4 * np.pi**2) / wi)

        # 3. Create the full scaling vector (shape (Nc*Nv*N_kpt,))
        # The eigenvectors are ordered such that for each band pair (c,v),
        # we have all k-points.
        scale_all = np.tile(scale, Nc * Nv)

        # 4. Reshape scale_all for broadcasting (shape (Nc*Nv*N_kpt, 1))
        scale_all = scale_all.reshape(-1, 1)

        # 5. Apply scaling element-wise
        A_scaled = A * scale_all

        # 6. Save the scaled complex array to a text file
        # Reshape the complex array into a 2D array where each complex number
        # becomes two columns (real, imag).
        # Output shape will be (Nc*Nv*N_kpt, NVAL * 2)
        
        A_reshaped_for_save = np.empty((A_scaled.shape[0], A_scaled.shape[1] * 2))
        A_reshaped_for_save[:, 0::2] = A_scaled.real
        A_reshaped_for_save[:, 1::2] = A_scaled.imag

        print(f"  Saving scaled eigenvectors to: {output_file_path}...")
        
        header_text = (f'Scaled Exciton Eigenvectors A. Shape: {A_scaled.shape}. '
                       f'Each complex value is two columns (real imag)')
        
        np.savetxt(
            output_file_path,
            A_reshaped_for_save,
            fmt='%.18e', # Format: scientific notation with high precision
            delimiter='  ', # Use two spaces as delimiter
            header=header_text,
            comments='# '
        )
        print("  Eigenvector saving complete.")

    except Exception as e:
        print(f"Error in save_A_output_txt: {e}", file=sys.stderr)
        print(f"Input shapes: A={A.shape}, wi={wi.shape}", file=sys.stderr)
        print(f"Parameters: Nc={Nc}, Nv={Nv}, N_kpt={N_kpt}", file=sys.stderr)
        sys.exit(1)

# Example Usage (optional - NOTE: This will write a file)
if __name__ == '__main__':
    # Create test data
    _Nc = 1
    _Nv = 1
    _N_kpt = 5
    _NVAL = 2
    _HBSE_size = _Nc * _Nv * _N_kpt

    _A_test = np.random.rand(_HBSE_size, _NVAL) + 1j * np.random.rand(_HBSE_size, _NVAL)
    _wi_test = np.abs(np.random.rand(_N_kpt)) + 0.1 # Ensure non-zero

    _output_dir = Path("./test_output")
    _output_dir.mkdir(exist_ok=True)
    _output_file = _output_dir / "A_test.txt"

    print("Testing save_A_output_txt...")
    print(f"  Input A shape: {_A_test.shape}")
    print(f"  Input wi shape: {_wi_test.shape}")

    save_A_output_txt(_output_file, _A_test, _Nc, _Nv, _wi_test, _N_kpt)

    if _output_file.exists():
        print(f"Test file '{_output_file}' created successfully.")
    else:
        print(f"Error: Test file '{_output_file}' was not created.", file=sys.stderr)
