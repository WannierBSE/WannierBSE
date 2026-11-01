function [Gi] = folding_G_vector(b1, b2, G_dir_max)
%==========================================================================
% FOLDING_G_VECTOR
%
% DESCRIPTION:
%   Returns the nearest-cell reciprocal vectors used to fold momentum transfers into the first Brillouin zone.
%
% INPUT ARGUMENTS:
%   b1, b2 : reciprocal lattice vectors.
%
% OUTPUT:
%   Gi : candidate reciprocal lattice shifts.
%==========================================================================

if nargin < 3, G_dir_max = []; end
Gi = direct_G_vectors(G_dir_max, b1, b2);

end
