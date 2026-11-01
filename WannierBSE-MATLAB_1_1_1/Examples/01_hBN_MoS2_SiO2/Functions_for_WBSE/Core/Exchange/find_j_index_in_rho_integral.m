function j_index_tot = find_j_index_in_rho_integral(R,tau,N_atoms,d_max,Basis_tau_index,Mmax_tolerance)
%==========================================================================
% FIND_J_INDEX_IN_RHO_INTEGRAL
%
% DESCRIPTION:
%   Builds allowed Wannier-index neighborhoods for rho integrals at a given lattice translation.
%
% INPUT ARGUMENTS:
%   R, tau : lattice translation and atomic positions.
%   N_atoms : number of atoms.
%   d_max, Mmax_tolerance : exchange distance and tolerance controls.
%   Basis_tau_index : basis-function indices grouped by atom.
%
% OUTPUT:
%   j_index_tot : basis-index cell array constrained by the mNN cutoff.
%==========================================================================


inplane_index = [1,2];

% Initialize as cell array to match project standard
j_index_tot = cell(length(Basis_tau_index), 1);

for Ni=1:N_atoms

    tau_p_index = Basis_tau_index{Ni};
    d_jjpR = sqrt(sum((R(inplane_index)+tau(:,inplane_index)-tau(Ni,inplane_index)).^2,2)); % calculate R+tau_{j}-tau_{j'}

    t_in = find(d_jjpR<=(d_max+Mmax_tolerance));
    L = 0;
    index_for_tau_jp = [];
    for ti = 1:length(t_in)
        Sn = length(Basis_tau_index{t_in(ti)});
        index_for_tau_jp((1:Sn)+L) = Basis_tau_index{t_in(ti)};
        L = length(index_for_tau_jp);
    end

    for ii=1:length(tau_p_index)
        j_index_tot{tau_p_index(ii)} = sort(index_for_tau_jp,'ascend');
    end

end

end
