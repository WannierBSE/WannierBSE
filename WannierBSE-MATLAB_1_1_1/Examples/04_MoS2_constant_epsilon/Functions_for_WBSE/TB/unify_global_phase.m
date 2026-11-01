function vec = unify_global_phase(vec)
%==========================================================================
% UNIFY_GLOBAL_PHASE
%
% DESCRIPTION:
%   Fixes the arbitrary global phase of a complex eigenvector using the phase of its coefficient sum.
%
% INPUT ARGUMENTS:
%   vec : complex state vector.
%
% OUTPUT:
%   vec : phase-normalized state vector.
%==========================================================================

    % Use the summed coefficient phase as a deterministic global gauge;
    % leave near-zero sums unchanged to avoid amplifying numerical noise.
    tol = 1.0e-14;
    s = sum(vec);
    if abs(s) > tol
        vec = vec * conj(s) / abs(s);
    end
end
