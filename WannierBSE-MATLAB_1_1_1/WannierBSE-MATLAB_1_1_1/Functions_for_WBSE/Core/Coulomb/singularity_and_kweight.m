function [Wq0, wi] = singularity_and_kweight(N_de, N_co, Nv_coarse, Nv_dense, d, ...
                                             q_data, epsQ2D_data, a, f)
%==========================================================================
% SINGULARITY_AND_KWEIGHT
%
% DESCRIPTION:
%   Computes q=0 singular-cell Coulomb averages and k-space weights for the mixed coarse/dense k-mesh.
%
% INPUT ARGUMENTS:
%   N_de, N_co : dense and coarse k-point counts.
%   Nv_coarse, Nv_dense : mesh resolutions.
%   d : effective material thickness.
%   a : lattice constant.
%   q_data, epsQ2D_data : dielectric screening table.
%
% OUTPUT:
%   Wq0 : singular Coulomb average per k-point.
%   wi : k-space integration weight per k-point.
%==========================================================================

[Wq0_co, wi_co] = coulomb_integral_d3h(Nv_coarse, a, d, q_data, epsQ2D_data);

% --- Dense Region Integration ---
[Wq0_de, wi_de] = coulomb_integral_d3h_dense(Nv_dense, a, d, q_data, epsQ2D_data, f);

% --- Global Vector Assembly ---
% Concatenate results for the full k-mesh (Dense followed by Coarse)
wi = [wi_de * ones(N_de, 1); wi_co * ones(N_co, 1)];
Wq0 = [Wq0_de * ones(N_de, 1); Wq0_co * ones(N_co, 1)];

end
