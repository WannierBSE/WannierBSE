function [Cc_tilde, Cv_tilde] = phase_tau_in_Vd(k, Cc, Cv, tau, WF_center)
%==========================================================================
% PHASE_TAU_IN_VD
%
% DESCRIPTION:
%   Applies atomic-center phase factors to TB coefficients for the long-range direct Coulomb approximation.
%
% INPUT ARGUMENTS:
%   k : k-point coordinates.
%   Cc, Cv : conduction and valence TB coefficients.
%   tau : atomic positions.
%   WF_center : Wannier-center mapping.
%
% OUTPUT:
%   Cc_tilde, Cv_tilde : phase-corrected coefficient arrays.
%==========================================================================

NB = size(WF_center, 1);
N_kpt = size(k, 1);
expiktau = zeros(N_kpt, NB);

% --- Phase Factor Computation ---
% Iterate over Wannier functions to determine atomic center and phase
for i = 1:NB
    tau_i = tau(WF_center(i, 2), :);
    expiktau(:, i) = exp(-1i * sum(bsxfun(@times, [k(:, 1), k(:, 2), zeros(N_kpt, 1)], tau_i), 2));
end

% --- Conduction Band Correction ---
Cc_tilde = zeros(size(Cc));
for n = 1:size(Cc, 3)
    for i = 1:size(Cc, 2)
        Cc_tilde(:, i, n) = Cc(:, i, n) .* expiktau(:, i);
    end
end

% --- Valence Band Correction ---
Cv_tilde = zeros(size(Cv));
for n = 1:size(Cv, 3)
    for i = 1:size(Cv, 2)
        Cv_tilde(:, i, n) = Cv(:, i, n) .* expiktau(:, i);
    end
end

end
