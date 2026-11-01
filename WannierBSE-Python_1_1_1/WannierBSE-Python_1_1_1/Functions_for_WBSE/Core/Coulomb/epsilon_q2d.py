"""
epsilon_q2d

PURPOSE:
    Interpolates the dielectric function epsilon(q) based on data.

DESCRIPTION:
    Python equivalent of epslion_Q2D.m, but more efficient.

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
from scipy.interpolate import interp1d

# --- Module-level cache for interpolators ---
# This ensures we only create the interpolator objects once,
# not for every single call to the function, which is much faster.
_interpolator_spline = None
_interpolator_linear = None
_min_q = None
_pre_max_q = None

def _initialize_interpolators(q_data, eps_q_2d_data):
    """
    _initialize_interpolators
    
    PURPOSE:
        Internal function to create and cache interpolators.
    
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
    global _interpolator_spline, _interpolator_linear, _min_q, _pre_max_q
    
    if q_data is None or eps_q_2d_data is None:
        raise ValueError("q_data or eps_q_2d_data is None in _initialize_interpolators")

    try:
        # Create the 'spline' interpolator for the main range
        _interpolator_spline = interp1d(
            q_data,
            eps_q_2d_data,
            kind='cubic', # 'cubic' is a good equivalent to MATLAB's 'spline'
            bounds_error=False,
            fill_value=np.nan # Will be handled by logic below
        )

        # Create the 'linear' interpolator for the end range
        _interpolator_linear = interp1d(
            q_data,
            eps_q_2d_data,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan # Will be handled by logic below
        )

        _min_q = q_data[0]
        _pre_max_q = q_data[-2] # Match MATLAB's (end-1)

        if _min_q < 0:
            print(f"Warning: q_data in dielectric file has negative values. min(q) = {_min_q}", file=sys.stderr)

    except Exception as e:
        print(f"Error: Failed to create interpolators in epsilon_q2d_py: {e}", file=sys.stderr)
        # Set to None to allow re-try (though it will likely fail again)
        _interpolator_spline = None
        _interpolator_linear = None
        raise # Re-raise the exception to stop execution

def epsilon_q2d_py(q_data, eps_q_2d_data, q):
    """
    epsilon_q2d_py
    
    PURPOSE:
        Python equivalent of epslion_Q2D.m.
    
    DESCRIPTION:
        Interpolates epsilon(q) based on provided data, using a cached
        interpolator.
    
    INPUT ARGUMENTS:
        q_data (np.ndarray): 1D array of q-points from the data file.
        eps_q_2d_data (np.ndarray): 1D array of corresponding epsilon
        values.
        q (float or np.ndarray): The q value(s) to interpolate at.
    
    OUTPUT:
        float or np.ndarray: The interpolated epsilon value(s).
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    global _interpolator_spline, _interpolator_linear, _min_q, _pre_max_q

    # Initialize interpolators on the first call
    if _interpolator_spline is None:
        _initialize_interpolators(q_data, eps_q_2d_data)

    # Make sure q is a numpy array for vectorized operations
    q_in = np.asarray(q)
    # Create output array with same shape as input q, initialized to 1.0
    epslnq = np.ones_like(q_in) # Default value is 1.0

    # Handle scalar input:
    is_scalar = q_in.ndim == 0
    if is_scalar:
        q_in = q_in.reshape(1) # Temporarily make it 1D array
        epslnq = np.ones(1)

    # --- Replicate MATLAB's logic for different ranges ---

    # 1. q < min(q_data)
    # This range is already set to 1.0 by np.ones_like()

    # 2. q >= min(q_data) AND q < q_data(end-1)
    t_middle = (q_in >= _min_q) & (q_in < _pre_max_q)
    if np.any(t_middle):
        epslnq[t_middle] = _interpolator_spline(q_in[t_middle])

    # 3. q >= q_data(end-1)
    t_larger = q_in >= _pre_max_q
    if np.any(t_larger):
        epslnq[t_larger] = _interpolator_linear(q_in[t_larger])

    # Handle any NaNs that might result from interpolation near boundaries
    if np.any(np.isnan(epslnq)):
        # print(f"Warning: NaN detected during epsilon interpolation. q={q_in[np.isnan(epslnq)]}. Defaulting to 1.0", file=sys.stderr)
        epslnq[np.isnan(epslnq)] = 1.0 # Default to 1 if interpolation fails

    if is_scalar:
        return epslnq.item() # Return scalar if input was scalar
    return epslnq
