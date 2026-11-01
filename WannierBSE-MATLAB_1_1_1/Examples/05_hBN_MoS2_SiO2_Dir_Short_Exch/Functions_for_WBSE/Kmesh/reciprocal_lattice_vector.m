function [b1,b2] = reciprocal_lattice_vector(a1,a2)
%==========================================================================
% RECIPROCAL_LATTICE_VECTOR
%
% DESCRIPTION:
%   Computes two-dimensional reciprocal lattice vectors from the in-plane real-space primitive vectors.
%
% INPUT ARGUMENTS:
%   a1, a2 : real-space primitive vectors.
%
% OUTPUT:
%   b1, b2 : reciprocal lattice vectors.
%==========================================================================


a1xa2_dot_a3 = a1(1)*a2(2)-a1(2)*a2(1);

b1 = (2*pi/a1xa2_dot_a3)*[a2(2),-a2(1),0];
b2 = (2*pi/a1xa2_dot_a3)*[-a1(2),a1(1),0];

b1 = [b1(1);b1(2)];
b2 = [b2(1);b2(2)];

end
