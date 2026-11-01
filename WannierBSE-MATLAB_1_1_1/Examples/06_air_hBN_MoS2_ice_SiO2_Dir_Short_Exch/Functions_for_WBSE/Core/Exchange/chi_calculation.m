function [chi] = chi_calculation( ...
    rho, R_mNN, G_all, kex, tau, N_atoms, d_max, Basis_tau_index, Mmax_tolerance, Npar_Xi, e, eps0, ...
    z_point_t, b1)
%==========================================================================
% CHI_CALCULATION
%
% DESCRIPTION:
%   Accumulates the short-range exchange tensor chi from pair-density tensors and spline-evaluated Xi integrals.
%
% INPUT ARGUMENTS:
%   rho, R_mNN, G_all : pair densities, retained translations, and G vectors.
%   kex, tau, N_atoms : exchange momentum, atomic positions, and atom count.
%   d_max, Basis_tau_index, Mmax_tolerance : exchange geometry controls.
%   Npar_Xi : maximum number of workers for Xi evaluation.
%   e, eps0, z_point_t, b1 : physical constants, z grid, and reciprocal vector.
%
% OUTPUT:
%   chi : exchange tensor over Wannier basis and retained lattice translations.
%==========================================================================


[hfz,hfz0] = spline_integral_z();
% Fixed-grid operator lifetime: construct once, before the G parfor.
w = xi_operator_build_w(hfz0,z_point_t);

% About the Xi, please refer to section 2.5 of " BSE_documentation.ppt " file for details.

%========= Xi and chi ==========
chi = complex(zeros(size(rho,1),size(rho,1),size(rho,1),size(rho,1),size(R_mNN,1),size(R_mNN,1)), ...
    zeros(size(rho,1),size(rho,1),size(rho,1),size(rho,1),size(R_mNN,1),size(R_mNN,1)));
if Npar_Xi > 1
    parfor (g=1:size(G_all,1),Npar_Xi)
        % One G operator lifetime: M operators are built once inside this worker
        % iteration; each R2 B batch is reused across all five R1 blocks.
        Xi_g = xi_operator_one_G(rho,R_mNN,G_all,g,kex,tau,N_atoms,d_max,...
            Basis_tau_index,Mmax_tolerance,e,eps0,z_point_t,b1,hfz,hfz0,w);
        chi = chi + Xi_g;
    end
else
    for g=1:size(G_all,1)
        Xi_g = xi_operator_one_G(rho,R_mNN,G_all,g,kex,tau,N_atoms,d_max,...
            Basis_tau_index,Mmax_tolerance,e,eps0,z_point_t,b1,hfz,hfz0,w);
        chi = chi + Xi_g;
    end
end
%========= Xi and chi ==========

end
