function [U,V,meta] = rho_operator_build_factors(G_all,kex,a1,a2,u_2D,u_point,v_point,kexTol,hu,hv,J)
%==========================================================================
% RHO_OPERATOR_BUILD_FACTORS
%
% DESCRIPTION:
%   Builds fixed separable rho plane-response factors. For fixed geometry
%   and G_all, rho(g) = U(g,:)*W*V(g,:).', where W is one in-plane Wij
%   array. The factors preserve the production scalar spline responses.
%
% INPUT ARGUMENTS:
%   G_all, kex : reciprocal vectors and exchange momentum.
%   a1, a2 : in-plane lattice vectors.
%   u_2D, u_point, v_point : in-plane grids.
%   kexTol, hu, hv, J : interpolation and integration controls.
%
% OUTPUT:
%   U, V : separable plane-response factors.
%   meta : construction metadata.
%==========================================================================

Nu = numel(u_point);
Nv = numel(v_point);
NG = size(G_all,1);
if Nu ~= Nv
    error('rho_operator_build_factors:NonSquareGrid', ...
          'The validated response construction requires the current square u/v grid.');
end
if numel(u_2D) ~= Nu*Nv
    error('rho_operator_build_factors:GridSize', ...
          'u_2D must contain exactly numel(u_point)*numel(v_point) entries.');
end

[hfuv,hfuv0] = spline_integral_inplane();
t_build = tic;
e1 = zeros(Nu,1);
e1(1) = 1;
A = complex(zeros(NG,Nu));
B = complex(zeros(NG,Nv));
for i = 1:Nu
    x = zeros(Nu,1);
    x(i) = 1;
    A(:,i) = rho_operator_scalar_all_g(x*e1.', hfuv, hfuv0, kex, a1, a2, ...
                                        u_2D, u_point, v_point, kexTol, hu, hv, J, G_all).';
end
for j = 1:Nv
    if j == 1
        B(:,j) = A(:,1);
    else
        y = zeros(Nv,1);
        y(j) = 1;
        B(:,j) = rho_operator_scalar_all_g(e1*y.', hfuv, hfuv0, kex, a1, a2, ...
                                            u_2D, u_point, v_point, kexTol, hu, hv, J, G_all).';
    end
end
anchor = A(:,1);
if any(abs(anchor) <= 1e-12)
    error('rho_operator_build_factors:ZeroAnchor', ...
          'The separable response anchor is too small for stable normalization.');
end
U = A ./ anchor;
V = B;
meta = struct('Nu',Nu, 'Nv',Nv, 'NG',NG, 'responses',Nu+Nv-1, ...
              'anchor_index',[1 1], 'construction_seconds',toc(t_build), ...
              'classU',class(U), 'classV',class(V), ...
              'complexU',~isreal(U), 'complexV',~isreal(V), ...
              'bytes',16*(numel(U)+numel(V)));
end

function out = rho_operator_scalar_all_g(W,hfuv,hfuv0,kex,a1,a2,u_2D,u_point,v_point,kexTol,hu,hv,J,G_all)
[c3,c2,c1,c0] = wf_spline_interpolation(W(:),kex,a1,a2,u_2D,u_point,v_point,kexTol);
out = complex(zeros(1,size(G_all,1)));
for g = 1:size(G_all,1)
    out(g) = rho_integral_v2(hfuv,hfuv0,c3,c2,c1,c0,kex,a1,a2,u_2D, ...
                             u_point,v_point,kexTol,hu,hv,J,G_all(g,:));
end
end
