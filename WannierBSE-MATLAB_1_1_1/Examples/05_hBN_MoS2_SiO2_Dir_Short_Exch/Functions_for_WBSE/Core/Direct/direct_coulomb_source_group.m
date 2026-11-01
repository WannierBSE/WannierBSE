function Vd_group = direct_coulomb_source_group( ...
    k, b1, b2, d, q_data, epsQ2D_data, Wq0_j, wi_j, tau, Basis_tau_index_cell, N_atoms, Cc_tilde, ...
    Cv_tilde, pairs, bands, j, e, eps0, G_dir_max)
%==========================================================================
% DIRECT_COULOMB_SOURCE_GROUP
%
% DESCRIPTION:
%   Evaluates all upper-triangular direct Coulomb blocks for one source
%   k-point. Source-dependent q folding, screening, and exp(iG*tau) factors
%   are reused across the requested band-pair blocks.
%
% INPUT ARGUMENTS:
%   k, b1, b2 : k-point and reciprocal-lattice coordinates.
%   d : effective material thickness.
%   q_data, epsQ2D_data : dielectric screening data.
%   Wq0_j, wi_j : source singular-cell value and integration weight.
%   tau, Basis_tau_index_cell, N_atoms : Wannier geometry and basis mapping.
%   Cc_tilde, Cv_tilde : phase-corrected conduction and valence coefficients.
%   pairs, bands, j : requested band pairs, band table, and source index.
%   e, eps0 : physical charge and vacuum permittivity constants.
%   G_dir_max : reciprocal-vector cutoff; omitted or empty defaults to 1.
%
% OUTPUT:
%   Vd_group : direct-interaction blocks for the requested source k-point.
%==========================================================================

if nargin < 19, G_dir_max = []; end
q = bsxfun(@minus, k, k(j, :));
[Vq, epsilon_q] = screened_coulomb_interaction(q, b1, b2, d, q_data, epsQ2D_data, G_dir_max);
[~, MinNdkG] = wave_vector_folding(q, b1, b2, G_dir_max);
expiGtau = direct_coulomb_expiGtau(MinNdkG, b1, b2, k, tau, Basis_tau_index_cell, N_atoms, size(Cc_tilde, 2), G_dir_max);
kernel = -((e^2) ./ (4 * pi * eps0)) .* (Vq ./ epsilon_q);

Vd_group = complex(zeros(size(k, 1), size(pairs, 1)), zeros(size(k, 1), size(pairs, 1)));
for p = 1:size(pairs, 1)
    base = pairs(p, 1);
    target = pairs(p, 2);
    vv = bands(base, 1);
    cc = bands(base, 2);
    v = bands(target, 1);
    c = bands(target, 2);

    Mc = sum((conj(Cc_tilde(:, :, c)) .* expiGtau) .* Cc_tilde(j, :, cc), 2);
    Mv = sum((conj(Cv_tilde(:, :, v)) .* expiGtau) .* Cv_tilde(j, :, vv), 2);
    Vd = kernel .* Mc(:) .* conj(Mv(:));
    Vd(j) = Vd_singularity(e, eps0, Wq0_j, wi_j, Cc_tilde, Cv_tilde, c, cc, v, vv, j);
    Vd_group(:, p) = Vd;
end
end
