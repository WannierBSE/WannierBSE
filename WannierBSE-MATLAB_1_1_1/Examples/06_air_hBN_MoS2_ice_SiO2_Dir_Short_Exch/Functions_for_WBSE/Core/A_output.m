function A_out = A_output(A, Nc, Nv, wi, N_kpt)
%==========================================================================
% A_OUTPUT
%
% DESCRIPTION:
%   Scales discretized exciton eigenvectors from the numerical BSE normalization to the
%   continuum-normalized convention used for post-processing.
%
% INPUT ARGUMENTS:
%   A : raw exciton eigenvector matrix ordered by band pair and k-point.
%   Nc, Nv : number of conduction and valence bands retained in the BSE basis.
%   wi : k-space integration weights for the generated momentum mesh.
%   N_kpt : total number of k-points.
%
% OUTPUT:
%   A_out : eigenvectors rescaled by the square-root k-weight factor.
%==========================================================================

scale = sqrt((4 * pi^2) ./ wi);                 % N_kpt x 1 vector

% Replicate scale across all band combinations (kron product)
scale_all = kron(ones(Nc * Nv, 1), scale);      % (Nc*Nv*N_kpt) x 1 vector

% Apply physical scaling
A_out = A .* scale_all;                 

end
