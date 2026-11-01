"""
reciprocal_lattice_vector

PURPOSE:
    Calculates the 2D reciprocal lattice vectors (b1, b2) from the

DESCRIPTION:
    real-space lattice vectors (a1, a2).
    Python equivalent of Reciprocal_lattice_vector.m.

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

def reciprocal_lattice_vector_py(a1, a2):
    """
    reciprocal_lattice_vector_py
    
    PURPOSE:
        Calculates the 2D reciprocal lattice vectors b1 and b2.
    
    DESCRIPTION:
        Implements the operation described by the function name and signature.
    
    INPUT ARGUMENTS:
        a1 (np.ndarray): Real-space lattice vector 1 (1D array/list, e.g., [ax, ay]).
        a2 (np.ndarray): Real-space lattice vector 2 (1D array/list, e.g., [ax, ay]).
        The z-component is ignored if present.
    
    OUTPUT:
        tuple: Contains the following elements:
        b1 (np.ndarray): 1D array (2 elements) for reciprocal vector b1.
        b2 (np.ndarray): 1D array (2 elements) for reciprocal vector b2.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    # Ensure inputs are NumPy arrays and extract 2D components
    try:
        a1_vec = np.asarray(a1)[:2] # Take only first 2 components
        a2_vec = np.asarray(a2)[:2] # Take only first 2 components

        if a1_vec.shape != (2,) or a2_vec.shape != (2,):
            raise ValueError("Input vectors must have at least 2 components.")

    except Exception as e:
        print(f"Error processing input vectors in reciprocal_lattice_vector_py: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate the 2D cross product (determinant for area)
    # Area = a1x * a2y - a1y * a2x
    area = a1_vec[0] * a2_vec[1] - a1_vec[1] * a2_vec[0]

    if abs(area) < 1e-12: # Check for near-zero area (collinear vectors)
        print("Error: Real-space lattice vectors a1 and a2 are collinear, cannot compute reciprocal vectors.", file=sys.stderr)
        sys.exit(1)

    # Calculate reciprocal vectors using the 2D formula
    # b1 = (2*pi / Area) * [ a2y, -a2x ]
    # b2 = (2*pi / Area) * [ -a1y,  a1x ]
    factor = (2 * np.pi) / area
    b1 = factor * np.array([a2_vec[1], -a2_vec[0]])
    b2 = factor * np.array([-a1_vec[1], a1_vec[0]])

    return b1, b2

# Example Usage (optional, runs only if script is executed directly)
if __name__ == '__main__':
    # Example lattice vectors (similar to hexagonal)
    a1_test = np.array([3.0, 0.0])
    a2_test = np.array([1.5, 3.0 * np.sqrt(3) / 2])

    print("Testing reciprocal_lattice_vector_py:")
    print(f"  a1 = {a1_test}")
    print(f"  a2 = {a2_test}")

    b1_test, b2_test = reciprocal_lattice_vector_py(a1_test, a2_test)

    print("\nCalculated Reciprocal Vectors:")
    print(f"  b1 = {b1_test}")
    print(f"  b2 = {b2_test}")

    # Verify orthogonality: a1 . b2 = 0, a2 . b1 = 0
    print("\nVerification (should be close to zero):")
    print(f"  a1 . b2 = {np.dot(a1_test, b2_test):.10f}")
    print(f"  a2 . b1 = {np.dot(a2_test, b1_test):.10f}")
    # Verify normalization: a1 . b1 = 2*pi, a2 . b2 = 2*pi
    print(f"\nVerification (should be close to 2*pi = {2*np.pi:.10f}):")
    print(f"  a1 . b1 = {np.dot(a1_test, b1_test):.10f}")
    print(f"  a2 . b2 = {np.dot(a2_test, b2_test):.10f}")
