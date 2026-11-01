function Gi = direct_G_vectors(G_dir_max, b1, b2)
%==========================================================================
% DIRECT_G_VECTORS
%
% DESCRIPTION:
%   Generates reciprocal-lattice vectors used by the direct Coulomb
%   interaction while preserving the zero vector.
%
% INPUT ARGUMENTS:
%   G_dir_max : reciprocal-vector cutoff; omitted or empty defaults to 1.
%   b1, b2 : reciprocal lattice vectors.
%
% OUTPUT:
%   Gi : candidate reciprocal lattice vectors.
%==========================================================================

if nargin < 1 || isempty(G_dir_max)
    G_dir_max = 1;
end
if ~isscalar(G_dir_max) || ~isfinite(G_dir_max) || G_dir_max < 0
    error('direct_G_vectors:InvalidCutoff', 'G_dir_max must be finite and nonnegative.');
end
b1 = reshape(b1, [1, 2]);
b2 = reshape(b2, [1, 2]);
if G_dir_max == 0
    Gi = [0, 0];
    return
end
limit = ceil(G_dir_max);
cutoff = 1.005 * G_dir_max * norm(b1);
Gi = zeros((2 * limit + 1)^2, 2);
keep = false(size(Gi, 1), 1);
row = 0;
for n = -limit:limit
    for m = -limit:limit
        row = row + 1;
        Gi(row, :) = m * b1 + n * b2;
        keep(row) = norm(Gi(row, :)) <= cutoff;
    end
end
Gi = Gi(keep, :);
end
