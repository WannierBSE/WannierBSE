function [hf, hf0] = SPint_uv
%==========================================================================
% SPINT_UV
%
% DESCRIPTION:
%   Returns analytic spline integration kernels for in-plane polynomial and oscillatory factors used in
%   Wannier Fourier integrals.
%
% INPUT ARGUMENTS:
%   None. The kernels are generated symbolically from fixed definitions.
%
% OUTPUT:
%   hf : oscillatory integral kernel.
%   hf0 : non-oscillatory polynomial integral kernel.
%==========================================================================

syms f(c1,c2,c3,c4,G,y0,h) f0(c1,c2,c3,c4,y0,h) y

% --- Analytical Integration ---

% 1. Integration with complex phase factor (Fourier-like)
f(c1,c2,c3,c4,G,y0,h) = int(((c1.*((y-y0).^3) + c2.*((y-y0).^2) + c3.*(y-y0) + c4) .* exp(-1i*G.*y)), y, [y0, y0+h]);

% 2. Standard polynomial integration (Normalization/Overlap)
f0(c1,c2,c3,c4,y0,h) = int(((c1.*((y-y0).^3) + c2.*((y-y0).^2) + c3.*(y-y0) + c4)), y, [y0, y0+h]);

% --- Optimization & Handle Generation ---
% Simplifying the expressions minimizes numerical overhead during large loops
f  = simplify(f);
f0 = simplify(f0);

% Convert to anonymous function handles for high-performance numerical calls
hf  = matlabFunction(f);
hf0 = matlabFunction(f0);

end
