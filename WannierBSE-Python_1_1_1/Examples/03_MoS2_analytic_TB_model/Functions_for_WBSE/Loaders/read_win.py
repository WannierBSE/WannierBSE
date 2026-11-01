"""
read_win

PURPOSE:
    Reads structural information (lattice vectors, atomic positions)

DESCRIPTION:
    from the 'structure.txt' file. Python equivalent of Read_win.m.

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

def resolve_structure_input_py(structure_file_path):
    """
    resolve_structure_input_py
    
    PURPOSE:
        Resolve structure source with WBSE priority: wannier90.win, then structure.txt.
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        See the annotated signature and detailed description.
    
    OUTPUT:
        See the annotated signature and detailed description.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    source = Path(structure_file_path)
    if source.is_dir():
        win_path = source / "wannier90.win"
        structure_path = source / "structure.txt"
    else:
        win_path = source.with_name("wannier90.win")
        structure_path = source
        if source.name == "wannier90.win":
            win_path = source
            structure_path = source.with_name("structure.txt")

    if win_path.is_file():
        return win_path, "User_input/wannier90.win"
    if structure_path.is_file():
        return structure_path, "User_input/structure.txt"
    raise FileNotFoundError(f"No structure input found. Expected {win_path} or {structure_path}.")


def read_win_py(structure_file_path):
    """
    read_win_py
    
    PURPOSE:
        Reads lattice vectors and atomic positions from the specified
    
    DESCRIPTION:
        structure file.
    
    INPUT ARGUMENTS:
        structure_file_path (str or Path): Path to the 'structure.txt'
        file.
    
    OUTPUT:
        tuple: Contains the following elements:
        a1 (np.ndarray): 1x3 lattice vector 1.
        a2 (np.ndarray): 1x3 lattice vector 2.
        a3 (np.ndarray): 1x3 lattice vector 3.
        a (float): Lattice constant (norm of a1).
        tau (np.ndarray): Nx3 array of atomic Cartesian coordinates.
        N_atoms (int): Number of atoms.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    a1, a2, a3, a = None, None, None, None
    tau = None
    N_atoms = 0
    lines = []

    try:
        structure_file_path, source_label = resolve_structure_input_py(structure_file_path)
        print(f"[INFO ] Structure extracted from {source_label}.")
        with open(structure_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Structure file not found at {structure_file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading structure file '{structure_file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Find lattice vector section
        start_lv = -1
        end_lv = -1
        for i, line in enumerate(lines):
            if 'begin unit_cell_cart' in line.strip().lower():
                start_lv = i
                break
        if start_lv >= 0:
            for i in range(start_lv + 1, len(lines)):
                if 'end unit_cell_cart' in lines[i].strip().lower():
                    end_lv = i
                    break
        if start_lv == -1 or end_lv == -1 or end_lv <= start_lv:
            raise ValueError("Could not find 'begin unit_cell_cart' section or not enough lines.")

        lattice_vectors = []
        for line in lines[start_lv + 1:end_lv]:
            line_strip = line.strip()
            if not line_strip:
                continue
            if line_strip.lower() == 'ang':
                continue
            if line_strip.lower() == 'bohr':
                raise ValueError("Unsupported unit declaration 'bohr'.")
            parts = line_strip.split()
            values = np.fromstring(' '.join(parts), sep=' ')
            if len(parts) != 3 or values.size != 3:
                raise ValueError("Lattice vectors do not have 3 components.")
            lattice_vectors.append(values)
            if len(lattice_vectors) > 3:
                raise ValueError("More than three lattice vectors found.")
        if len(lattice_vectors) != 3:
            raise ValueError("Lattice vectors do not have 3 components.")
        a1, a2, a3 = lattice_vectors

        a = np.linalg.norm(a1) # Calculate lattice constant

        # Find atomic positions section
        start_atoms = -1
        end_atoms = -1
        atom_block_type = None
        for i, line in enumerate(lines):
            line_strip = line.strip().lower()
            if 'begin atoms_cart' in line_strip:
                start_atoms = i + 1
                atom_block_type = 'cart'
                break
            elif 'begin atoms_frac' in line_strip:
                start_atoms = i + 1
                atom_block_type = 'frac'
                break

        if atom_block_type == 'cart':
            end_marker = 'end atoms_cart'
        elif atom_block_type == 'frac':
            end_marker = 'end atoms_frac'
        else:
            end_marker = None
        for i in range(start_atoms, len(lines)) if start_atoms >= 0 else []:
            if end_marker in lines[i].strip().lower():
                end_atoms = i
                break

        if start_atoms == -1 or end_atoms == -1 or start_atoms >= end_atoms:
            raise ValueError("Could not find valid 'begin atoms_cart'/'end atoms_cart' section.")

        N_atoms = end_atoms - start_atoms
        if N_atoms <= 0:
            raise ValueError("No atoms found between atom section markers.")

        atom_lines = lines[start_atoms:end_atoms]
        tau_list = []
        for i, line in enumerate(atom_lines):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Atom line {start_atoms + i} has too few columns: '{line.strip()}'")
            
            # Assumes format: AtomSymbol x y z ...
            # We take columns 1, 2, 3 (0-indexed) as coordinates
            coords = np.fromstring(' '.join(parts[1:4]), sep=' ')
            if coords.size != 3:
                raise ValueError(f"Could not parse 3 coordinates from atom line {start_atoms + i}: '{line.strip()}'")
            tau_list.append(coords)
        
        tau = np.array(tau_list)
        if atom_block_type == 'frac':
            tau = tau @ np.vstack((a1, a2, a3))

    except ValueError as e:
        print(f"Error parsing structure file '{structure_file_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except IndexError:
        print(f"Error parsing structure file '{structure_file_path}': Reached end of file unexpectedly.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}", file=sys.stderr)
        sys.exit(1)

    # Return values consistent with MATLAB version
    return a1, a2, a3, a, tau, N_atoms

# Example Usage (optional)
if __name__ == '__main__':
    # Assume 'structure.txt' is in '../parameters/' relative to this script
    script_dir = Path(__file__).parent
    structure_file = script_dir.parent / 'parameters' / 'structure.txt'
    if structure_file.exists():
        print(f"Testing read_win_py with: {structure_file}")
        a1_test, a2_test, a3_test, a_test, tau_test, n_atoms_test = read_win_py(structure_file)
        print("Structure data read successfully:")
        print(f"  a1 = {a1_test}")
        print(f"  a2 = {a2_test}")
        print(f"  a3 = {a3_test}")
        print(f"  a = {a_test:.4f}")
        print(f"  N_atoms = {n_atoms_test}")
        print(f"  tau =\n{tau_test}")
    else:
        print(f"Test structure file not found at {structure_file}")
