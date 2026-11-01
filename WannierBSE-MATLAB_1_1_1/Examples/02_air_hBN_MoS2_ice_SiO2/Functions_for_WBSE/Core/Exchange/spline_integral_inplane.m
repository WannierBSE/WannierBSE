function [hf,hf0] = spline_integral_inplane()
%==========================================================================
% SPLINE_INTEGRAL_INPLANE
%
% DESCRIPTION:
%   Returns analytic cubic-spline integral kernels for in-plane Fourier factors.
%
% INPUT ARGUMENTS:
%   None. The kernels are generated symbolically from their fixed definitions.
%
% OUTPUT:
%   hf, hf0 : function handles for oscillatory and non-oscillatory spline integrals.
%==========================================================================

    syms f(c3,c2,c1,c0,G,y0,h) f0(c3,c2,c1,c0,y0,h) y y0
    
    % Kernel with momentum
    f(c3,c2,c1,c0,G,y0,h) = int(((c3.*((y-y0).^3)+c2.*((y-y0).^2)+c1.*(y-y0)+c0).*exp(-1i*G.*y)),y,[y0,y0+h]);
    
    % Kernel without momentum (Removed G from signature)
    f0(c3,c2,c1,c0,y0,h) = int(((c3.*((y-y0).^3)+c2.*((y-y0).^2)+c1.*(y-y0)+c0)),y,[y0,y0+h]);
    
    f = simplify(f);
    f0 = simplify(f0);
    
    hf = matlabFunction(f);
    hf0 = matlabFunction(f0);
end
