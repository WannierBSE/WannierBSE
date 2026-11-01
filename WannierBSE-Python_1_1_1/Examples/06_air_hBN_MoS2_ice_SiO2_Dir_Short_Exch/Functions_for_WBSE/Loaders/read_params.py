"""
read_params

PURPOSE:
    Reads calculation parameters from a control file (e.g., 'control.txt').

DESCRIPTION:
    Handles key=value pairs, ignores comments (starting with '%'),
    removes inline comments, and attempts to convert values to numbers
    (int or float) or numpy arrays (for vectors). Falls back to string.
    Removes trailing semicolons from values.

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

def read_params_py(fname):
    """
    read_params_py
    
    PURPOSE:
        Reads parameters from the specified file.
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        fname (str or Path): Path to the parameter file.
    
    OUTPUT:
        dict: A dictionary containing the parameters.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    params = {}
    line_content = "" # Keep track of current line for error reporting
    try:
        with open(fname, 'r') as f:
            for line in f:
                line_content = line # Store current line
                line = line.strip()
                if not line or line.startswith('%'):
                    continue

                # Remove inline comments
                if '%' in line:
                    line = line.split('%', 1)[0].strip()

                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip()
                    # Remove potential semicolons BEFORE parsing
                    val = val.strip().replace(';', '')

                    # Try to convert to numeric (scalar or vector)
                    try:
                        # Use numpy's fromstring for array capability for vectors/scalars.
                        # Non-numeric text such as true/false can produce an empty array
                        # instead of raising, so keep it as text in that case.
                        num_val = np.fromstring(val, sep=' ')
                        if num_val.size == 0:
                            params[key] = val
                        elif num_val.size == 1:
                            # Store as scalar if it's a single number
                            # Use item() to get Python native int/float
                            params[key] = num_val.item()
                        else:
                            params[key] = num_val # Store as numpy array
                    except ValueError:
                        # Store as string if not numeric
                        params[key] = val

    except FileNotFoundError:
        print(f"Error: Control file not found at {fname}", file=sys.stderr)
        sys.exit(1) # Use sys.exit for cleaner termination
    except Exception as e:
        print(f"Error reading parameters from '{fname}': {e}", file=sys.stderr)
        print(f"Problematic line might be: {line_content.strip()}", file=sys.stderr)
        sys.exit(1)

    return params

# Example Usage (optional, runs only if this script is executed directly)
if __name__ == '__main__':
    # Assuming 'control.txt' is in a 'parameters' subdirectory relative to this script
    from pathlib import Path
    script_dir = Path(__file__).parent
    param_file = script_dir / 'parameters' / 'control.txt'
    if param_file.exists():
        print(f"Testing read_params_py with: {param_file}")
        parameters = read_params_py(param_file)
        print("Parameters read successfully:")
        for k, v in parameters.items():
            print(f"  {k} = {v} (Type: {type(v).__name__})")
    else:
        print(f"Test parameter file not found at {param_file}")
