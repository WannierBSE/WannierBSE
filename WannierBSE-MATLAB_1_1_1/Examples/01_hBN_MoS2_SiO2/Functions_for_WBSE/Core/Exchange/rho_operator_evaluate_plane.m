function rho_G = rho_operator_evaluate_plane(U,V,W)
%==========================================================================
% RHO_OPERATOR_EVALUATE_PLANE
%
% DESCRIPTION:
%   Evaluates all rho G channels for one Wannier-pair plane using direct
%   non-conjugating bilinear contractions in the incoming G order.
%
% INPUT ARGUMENTS:
%   U, V : separable plane-response factors.
%   W : in-plane Wannier-pair response matrix.
%
% OUTPUT:
%   rho_G : row vector of Fourier-channel pair-density responses.
%==========================================================================

if ~isequal(size(U),size(V)) || size(W,1) ~= size(U,2) || size(W,2) ~= size(V,2)
    error('rho_operator_evaluate_plane:DimensionMismatch', ...
          'U, V, and W dimensions are incompatible for rho evaluation.');
end
NG = size(U,1);
rho_G = complex(zeros(1,NG));
for g = 1:NG
    rho_G(g) = U(g,:) * W * V(g,:).';
end
end
