function [J, hu, hv, hz, u, v, u_point, v_point, z_point] = CoordinateTransformation(r, a1, a2, Nz)
%==========================================================================
% COORDINATETRANSFORMATION
%
% DESCRIPTION:
%   Transforms Cartesian Wannier grids to oblique lattice coordinates and returns integration metrics
%   for spline quadrature.
%
% INPUT ARGUMENTS:
%   r : Cartesian grid coordinates.
%   a1, a2 : in-plane lattice vectors.
%   Nz : number of z-grid points.
%
% OUTPUT:
%   J : Jacobian factor.
%   hu, hv, hz : grid spacings.
%   u, v : transformed coordinates.
%   u_point, v_point, z_point : sorted grid coordinates.
%==========================================================================

a1x = a1(1); a1y = a1(2);
a2x = a2(1); a2y = a2(2);

% --- Extract Cartesian Coordinates ---
x = r(:,1);
y = r(:,2);
z = r(:,3);

% --- Coordinate Mapping (Oblique Transformation) ---
u = x - (a2x/a2y) .* y;
v = -x + (a1x/a1y) .* y;

% --- Grid Point Identification ---
% Using uniquetol to handle floating-point precision in grid alignment
Tol = 1e-6;
u_point = uniquetol(u, Tol);
v_point = uniquetol(v, Tol);
z_point = transpose(linspace(min(z), max(z), Nz));

% --- Differential Spacings ---
hu = u_point(2) - u_point(1);
hv = v_point(2) - v_point(1);
hz = (max(z) - min(z)) / (Nz - 1);

% --- Jacobian Calculation ---
% Represents the area transformation for the integration measure
Beta = (-a2x * a1y) + (a1x * a2y);
J = ((a1x * a1y * (a2y^2)) - (a2x * (a1y^2) * a2y)) / (Beta^2);

end
