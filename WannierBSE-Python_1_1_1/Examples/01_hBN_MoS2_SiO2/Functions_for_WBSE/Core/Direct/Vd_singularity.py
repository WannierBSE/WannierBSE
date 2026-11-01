"""
Vd_singularity

PURPOSE:
    Calculates the Vd matrix element for the singularity at q=0 (k=k').

DESCRIPTION:
    Python equivalent of Vd_singularity.m.

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
import traceback

def Vd_singularity_py(
    e, eps0, Wq0_j, wi_j, Cc_kex_tilde, Cv_tilde, c, cc, v, vv, j,
    verbose=False
):
    """
    Vd_singularity_py
    
    PURPOSE:
        Calculates the Vd matrix element for the diagonal k=k' singularity.
    
    DESCRIPTION:
        This function *exactly* matches the logic in the original
        Vd_singularity.m.
    
    INPUT ARGUMENTS:
        (all args): All necessary data pre-loaded by BSE.py.
        j (int): The 0-based k-point index.
        c, cc, v, vv (int): 1-based band indices.
        verbose (bool): If True, print debug info.
    
    OUTPUT:
        complex: The scaled Vd_kk element.
    
    PHYSICS/NUMERICAL NOTES:
        Preserves the numerical conventions, units, indexing, and array ordering established by the caller.
    
    SIDE EFFECTS:
        None unless explicitly documented by this function.
    """
    if verbose:
        print(f"  [Debug] Vd_singularity_py: Calculating for k-point j={j}...")

    try:
        # Note: Cc_kex_tilde and Cv_tilde are full (N_kpt, NB, N_bands) 
        # arrays. We only need the slice at k-point 'j'.

        # --- MATLAB Logic: ---
        # # Mc = sum(conj(Cc_kex_tilde(j,:,c)) .* Cc_kex_tilde(j,:,cc))
        # # Mv = sum(      Cv_tilde(j,:,v)    .* conj(Cv_tilde(j,:,vv))) 
        # Note the conjugate order
        # # Vd_kk = -const * Wq0 * Mc * Mv                      <-- 
        # Note NO conjugate on Mv
        # # Vd_kk = Vd_kk / wi

        # 1. Get the slices for k-point 'j'
        # C_tilde_j_c  = Cc_kex_tilde[j, :, c-1] # 0-based band index
        C_tilde_j_cc = Cc_kex_tilde[j, :, cc-1]
        C_tilde_j_c  = Cc_kex_tilde[j, :, c-1]
        
        C_tilde_j_v  = Cv_tilde[j, :, v-1]
        C_tilde_j_vv = Cv_tilde[j, :, vv-1]

        # 2. Calculate Mc_overlap (matches MATLAB)
        # Mc_overlap = sum(conj(C_tilde_j_c) * C_tilde_j_cc)
        Mc_overlap = np.sum(np.conj(C_tilde_j_c) * C_tilde_j_cc)

        # 3. Calculate Mv_overlap (This is the fix to match MATLAB's 
        # specific logic)
        # Mv_overlap = sum(C_tilde_j_v * conj(C_tilde_j_vv))
        Mv_overlap = np.sum(C_tilde_j_v * np.conj(C_tilde_j_vv))

        # 4. Calculate Vd_kk (unweighted singularity term)
        # We MUST divide by wi_j to match the MATLAB implementation's "average" interaction convention.
        Vd_kk = -((e**2) / (4 * np.pi * eps0)) * Wq0_j * Mc_overlap * Mv_overlap
        Vd_kk = Vd_kk / wi_j

        if verbose:
            print(f"  [Debug] Vd_singularity_py: k={j}, Wq0={Wq0_j:.4e}, wi={wi_j:.4e}, "
                  f"Mc={Mc_overlap:.4f}, Mv={Mv_overlap:.4f}, Vd_kk={Vd_kk:.4f}")

        return Vd_kk

    except Exception as e:
        print(f"--- ERROR in Vd_singularity_py ---", file=sys.stderr)
        print(f"Inputs: j={j}, c={c}, cc={cc}, v={v}, vv={vv}", file=sys.stderr)
        print(f"Cc_kex_tilde={Cc_kex_tilde.shape}, Cv_tilde={Cv_tilde.shape}", file=sys.stderr)
        print(f"Exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
