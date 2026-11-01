"""
phys_constant

PURPOSE:
    Reads physical constants from a .dat file and converts them to

DESCRIPTION:
    atomic units used by the BSE calculation.
    Python equivalent of phys_constant.m.

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

try:
    # --- MODIFICATION: Use relative import ---
    # The '.' tells Python to look in the current package 
    # (Functions_for_WBSE/Utils)
    from Functions_for_WBSE.Loaders.read_params import read_params_py
    # --- END MODIFICATION ---
except ImportError as e:
    print(f"Error: Could not import 'read_params_py'.", file=sys.stderr)
    print(f"Ensure 'read_params.py' is in the same directory as 'phys_constant.py'.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)

def phys_constant_py(constants_file_path):
    """
    phys_constant_py
    
    PURPOSE:
        Reads SI-unit constants from a file and returns values
    
    DESCRIPTION:
        converted for the calculation.
    
    INPUT ARGUMENTS:
        constants_file_path (str or Path): Path to the
        'physical_constants.dat' file.
    
    OUTPUT:
        tuple: Contains the following elements:
        e (float): Elementary charge, normalized (1.0).
        eps0 (float): Converted vacuum permittivity in
        (e^2)/(eV*Angstrom).
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    try:
        # Read the raw SI-unit constants from the file
        raw_constants = read_params_py(constants_file_path)

        # Extract the necessary values
        e_macro = raw_constants['e']
        eps0_macro = raw_constants['eps0']

    except FileNotFoundError:
        print(f"Error: Physical constants file not found at {constants_file_path}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Constant {e} not found in {constants_file_path}.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing physical constants: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Perform Unit Conversions ---
    # This logic is identical to the original phys_constant.m

    # 1. 'e' is normalized to 1.0 (unit of charge)
    e = e_macro / e_macro

    # 2. Convert eps0 units from F/m to (e^2) / (eV * Angstrom)
    #    The unit conversion is based on the original MATLAB code:
    #    eps0 = eps0_macro * (1 / (e_macro * 1e10))
    eps0 = eps0_macro * (1.0 / (e_macro * 1e10))

    return e, eps0

# Example Usage (optional)
if __name__ == '__main__':
    # --- MODIFICATION: PATH ---
    # Look for the constants file in the *same* directory as this script
    script_dir = Path(__file__).parent
    constants_file = script_dir / 'physical_constants.dat'
    # --- END MODIFICATION ---

    if constants_file.exists():
        print(f"Testing phys_constant_py with: {constants_file}")
        e_test, eps0_test = phys_constant_py(constants_file)
        print("Physical constants read and converted successfully:")
        print(f"e (normalized) = {e_test}")
        print(f"eps0 (converted) = {eps0_test}")
    else:
        print(f"Test constants file not found at {constants_file}")
