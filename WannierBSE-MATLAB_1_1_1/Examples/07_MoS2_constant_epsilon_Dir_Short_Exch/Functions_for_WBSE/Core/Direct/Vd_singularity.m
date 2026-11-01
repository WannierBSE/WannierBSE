function [Vd_kk] = Vd_singularity(e, eps0, Wq0, wi, Cc_kex_tilde, Cv_tilde, c, cc, v, vv, i)
%==========================================================================
% VD_SINGULARITY
%
% DESCRIPTION:
%   Evaluates the pre-integrated q=0 direct Coulomb singularity correction for one BSE matrix element.
%
% INPUT ARGUMENTS:
%   e, eps0 : WBSE unit physical constants.
%   Wq0 : averaged screened Coulomb value over the singular k-cell.
%   wi : integration weight for the target k-point.
%   Cc_kex_tilde, Cv_tilde : phase-corrected conduction and valence TB coefficients.
%   c, cc, v, vv : conduction and valence band indices.
%   i : k-point index.
%
% OUTPUT:
%   Vd_kk : singular direct-interaction contribution normalized by wi.
%==========================================================================

Vd_kk = -(((e^2) ./ (4 * pi * eps0)) .* (Wq0) ...
        .* sum(conj(Cc_kex_tilde(i, :, c)) .* Cc_kex_tilde(i, :, cc)) ...
        .* sum(Cv_tilde(i, :, v) .* conj(Cv_tilde(i, :, vv)))); 

% Normalize by the integration weight
Vd_kk = Vd_kk / wi;

end
