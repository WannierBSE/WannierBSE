function [Vkkp, delta_kV] = coulomb_integral_d3h(Nt, a, d, q_data, epsQ2D_data)
%==========================================================================
% COULOMB_INTEGRAL_D3H
%
% DESCRIPTION:
%   Integrates the z-averaged screened Coulomb kernel over one coarse D3h k-cell around the singular point.
%
% INPUT ARGUMENTS:
%   Nt : coarse mesh sampling parameter.
%   a : lattice constant in Angstrom.
%   d : effective material thickness.
%   q_data, epsQ2D_data : tabulated dielectric screening model.
%
% OUTPUT:
%   Vkkp : cell-averaged screened Coulomb integral.
%   delta_kV : associated k-space integration area.
%==========================================================================

Vq = @(kx,ky) bare_coulomb_q2d(sqrt(kx.^2 + ky.^2), d);
epsilon_q = @(kx,ky) (epsilon_Q2D(q_data, epsQ2D_data, sqrt(kx.^2 + ky.^2)));

% Define the integrand: V(q) / epsilon(q)
f = @(kx, ky) Vq(kx, ky) ./ epsilon_q(kx, ky);

% --- Geometric Grid Definitions ---
v1 = (pi/a).*[2/3, 2/sqrt(3)];
v1_unit = v1./Nt;
Nv1_unit = norm(v1_unit);
h = Nv1_unit*(sqrt(3)/2);
l1 = Nv1_unit/sqrt(3);
l2 = Nv1_unit/(2*sqrt(3));

% Integration Tolerance Settings
AbsTol = 1e-7;
RelTol = 5*1e-4;
m = -sqrt(3); % Geometric slope factor

% --- REGION 1: Rectangular Sub-domain ---
rec_kxmin = -l2*(sqrt(3)/2);
rec_kxmax = l2*(sqrt(3)/2);
rec_kymin = -l2;
rec_kymax = l2/2;

Vkkp_rec_iterated1 = integral2(f,rec_kxmin,0,0,rec_kymax,'method','iterated','AbsTol',AbsTol,'RelTol',RelTol);
Vkkp_rec_iterated2 = integral2(f,0,rec_kxmax,0,rec_kymax,'method','iterated','AbsTol',AbsTol,'RelTol',RelTol);
Vkkp_rec_iterated3 = integral2(f,rec_kxmin,0,rec_kymin,0,'method','iterated','AbsTol',AbsTol,'RelTol',RelTol);
Vkkp_rec_iterated4 = integral2(f,0,rec_kxmax,rec_kymin,0,'method','iterated','AbsTol',AbsTol,'RelTol',RelTol);

Vkkp_rec_iterated = Vkkp_rec_iterated1+Vkkp_rec_iterated2+Vkkp_rec_iterated3+Vkkp_rec_iterated4;

% --- REGION 2: Left Triangular Sub-domain ---
LT_kxmin = -Nv1_unit*cos(pi/3);
LT_kxmax = -l2*(sqrt(3)/2);
LT_kymin = @(kx) -l2+0*kx;
LT_kymax = @(kx) l1+m.*abs(kx);

Vkkp_LT_iterated = integral2(f,LT_kxmin,LT_kxmax,LT_kymin,LT_kymax,'method','iterated','AbsTol',1e-13,'RelTol',1e-10);

% --- REGION 3: Right Triangular Sub-domain ---
RT_kxmin = l2*(sqrt(3)/2);
RT_kxmax = Nv1_unit*cos(pi/3);
RT_kymin = @(kx) -l2+0*kx;
RT_kymax = @(kx) l1+m.*abs(kx);

Vkkp_RT_iterated = integral2(f,RT_kxmin,RT_kxmax,RT_kymin,RT_kymax,'method','iterated','AbsTol',1e-13,'RelTol',1e-10);

% --- REGION 4: Top Triangular Sub-domain ---
TT_kxmin = -l2*(sqrt(3)/2);
TT_kxmax = l2*(sqrt(3)/2);
TT_kymin = @(kx) (l2/2)+0*kx;
TT_kymax = @(kx) l1+m.*abs(kx);

Vkkp_TT_iterated = integral2(f,TT_kxmin,TT_kxmax,TT_kymin,TT_kymax,'method','iterated','AbsTol',1e-13,'RelTol',1e-10);

% --- Global Summation and Normalization ---
Vkkp = Vkkp_rec_iterated + Vkkp_LT_iterated + Vkkp_RT_iterated + Vkkp_TT_iterated;

% Unit area of the sampling parallelogram
delta_kV = h * Nv1_unit / 2; 

end
