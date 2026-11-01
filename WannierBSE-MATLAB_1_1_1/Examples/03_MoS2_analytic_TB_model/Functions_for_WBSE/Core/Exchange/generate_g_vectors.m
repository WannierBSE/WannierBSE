function [G_all] = generate_g_vectors(Gmax,b1,b2)
%==========================================================================
% GENERATE_G_VECTORS
%
% DESCRIPTION:
%   Generates nonzero reciprocal lattice vectors with magnitude below Gmax for short-range exchange.
%
% INPUT ARGUMENTS:
%   Gmax : reciprocal-vector magnitude cutoff.
%   b1, b2 : reciprocal lattice vectors.
%
% OUTPUT:
%   G_all : retained nonzero G vectors.
%==========================================================================

G_all = g_vector_v2(Gmax,b1,b2);

% Always exclude G=0 term (Strictly Short Range)
tG0 = find(G_all(:,1)==0 & G_all(:,2)==0);
G_all(tG0,:) = [];

end
