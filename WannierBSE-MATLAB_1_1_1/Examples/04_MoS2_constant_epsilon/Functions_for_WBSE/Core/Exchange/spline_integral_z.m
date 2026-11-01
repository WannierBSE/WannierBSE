function [hf,hf0] = spline_integral_z()
%==========================================================================
% SPLINE_INTEGRAL_Z
%
% DESCRIPTION:
%   Returns analytic cubic-spline integral kernels for z-direction exchange factors.
%
% INPUT ARGUMENTS:
%   None. The kernels are generated symbolically from their fixed definitions.
%
% OUTPUT:
%   hf, hf0 : function handles for exponential and polynomial spline integrals.
%==========================================================================

    syms f(c3,c2,c1,c0,G,y0,h,yc) f0(c3,c2,c1,c0,G,y0,h) y y0 yc
    
    % Analytical integral of (Cubic Spline Segment) * exp(-G*(z-zc))
    f(c3,c2,c1,c0,G,y0,h,yc) = int(((c3.*((y-y0).^3)+c2.*((y-y0).^2)+c1.*(y-y0)+c0).*exp(-G.*(y-yc))),y,[y0,y0+h]);
    
    % Analytical integral of (Cubic Spline Segment) only
    f0(c3,c2,c1,c0,G,y0,h) = int(((c3.*((y-y0).^3)+c2.*((y-y0).^2)+c1.*(y-y0)+c0)),y,[y0,y0+h]);
    
    f = simplify(f);
    f0 = simplify(f0);
    
    hf = matlabFunction(f);
    hf0 = matlabFunction(f0);
end
