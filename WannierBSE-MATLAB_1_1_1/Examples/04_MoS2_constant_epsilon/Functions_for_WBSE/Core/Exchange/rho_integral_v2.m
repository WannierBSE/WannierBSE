function [rho_z] = rho_integral_v2(hf,hf0,c3,c2,c1,c0,kex,a1,a2,u_2D,u_point_t,v_point_t,kexTol,hu,hv,J,G)
%==========================================================================
% RHO_INTEGRAL_V2
%
% DESCRIPTION:
%   Evaluates one spline-based in-plane pair-density Fourier integral for selected Wannier-index
%   neighborhoods.
%
% INPUT ARGUMENTS:
%   hf, hf0 : spline integration function handles.
%   rho coefficients, kex, lattice vectors, grids, and tolerances : integral data.
%   G : reciprocal vector for the Fourier channel.
%
% OUTPUT:
%   rho_z : z-resolved pair-density integral.
%==========================================================================


Nu = length(u_point_t);
Nv = length(v_point_t);

a1x = a1(1); a1y = a1(2);
a2x = a2(1); a2y = a2(2);
Beta = (-a2x*a1y)+(a1x*a2y);

kex_uu = ((kex(1)*a1x*a2y)+(kex(2)*a1y*a2y))./Beta;
kex_vv = ((kex(1)*a2x*a1y)+(kex(2)*a1y*a2y))./Beta;

if ( norm(kex_uu)<kexTol && norm(kex_vv)>kexTol )
    kexr_u = sum(bsxfun(@times,u_2D,kex_uu),2);
    kexr_v = zeros(size(v_point_t,1),1);
elseif ( norm(kex_uu)>kexTol && norm(kex_vv)<kexTol )
    kexr_u = zeros(size(u_2D,1),1);
    kexr_v = sum(bsxfun(@times,v_point_t,kex_vv),2);
elseif ( norm(kex_uu)<kexTol && norm(kex_vv)<kexTol )
    kexr_u = sum(bsxfun(@times,u_2D,kex_uu),2);
    kexr_v = sum(bsxfun(@times,v_point_t,kex_vv),2);
else
    kexr_u = zeros(size(u_2D,1),1);
    kexr_v = zeros(size(v_point_t,1),1);
end

% Momentum factors
Guu = ((G(1)*a1x*a2y)+(G(2)*a1y*a2y))./Beta;
Gvv = ((G(1)*a2x*a1y)+(G(2)*a1y*a2y))./Beta;

if ( norm(kex_uu)<kexTol && norm(kex_vv)>kexTol )
    Gvv = Gvv+kex_vv;
elseif ( norm(kex_uu)>kexTol && norm(kex_vv)<kexTol )
    Guu = Guu+kex_uu;
elseif ( norm(kex_uu)<kexTol && norm(kex_vv)<kexTol )
    % No shift
else
    Guu = Guu+kex_uu;
    Gvv = Gvv+kex_vv;
end

%---------------int u------------------
if abs(Guu)<kexTol
    Su = J * sum(hf0(c3, c2, c1, c0, u_point_t(1:end-1), hu), 1);
else
    Su = J * sum(hf(c3, c2, c1, c0, Guu, u_point_t(1:end-1), hu), 1);
end
Su = transpose(squeeze(Su));

%---------------int v------------------
SPv = spline(v_point_t, transpose(squeeze(Su).*exp(-1i*kexr_v)));
cv = SPv.coefs;
% cv order: c3, c2, c1, c0

if abs(Gvv)<kexTol
    rho_z = transpose(sum(hf0(cv(:,1), cv(:,2), cv(:,3), cv(:,4), v_point_t(1:end-1), hv), 1));
else
    rho_z = transpose(sum(hf(cv(:,1), cv(:,2), cv(:,3), cv(:,4), Gvv, v_point_t(1:end-1), hv), 1));
end

end
