function expiGtau = direct_coulomb_expiGtau(MinNdkG, b1, b2, k, tau, Basis_tau_index_cell, N_atoms, NB, G_dir_max)
%==========================================================================
% DIRECT_COULOMB_EXPIGTAU
%
% DESCRIPTION:
%   Builds the exp(iG*tau) phase factors for reciprocal vectors selected by
%   the folded direct-Coulomb momentum transfer.
%
% INPUT ARGUMENTS:
%   MinNdkG : selected reciprocal-vector indices for each momentum transfer.
%   b1, b2 : reciprocal lattice vectors.
%   k : k-point coordinates.
%   tau : atomic positions.
%   Basis_tau_index_cell : basis-function indices grouped by atom.
%   N_atoms : number of atoms.
%   NB : number of basis functions.
%   G_dir_max : reciprocal-vector cutoff; omitted or empty defaults to 1.
%
% OUTPUT:
%   expiGtau : phase-factor matrix for direct-Coulomb contractions.
%==========================================================================

if nargin < 9, G_dir_max = []; end
Gi = folding_G_vector(b1, b2, G_dir_max);
Gkkp = Gi(MinNdkG, :);
expiGtau = zeros(size(k, 1), NB);
for Ni = 1:N_atoms
    atom_phase = exp(1i * sum(bsxfun(@times, [Gkkp(:, 1), Gkkp(:, 2), zeros(size(k, 1), 1)], tau(Ni, :)), 2));
    expiGtau(:, Basis_tau_index_cell{Ni}) = repmat(atom_phase, 1, length(Basis_tau_index_cell{Ni}));
end
end
