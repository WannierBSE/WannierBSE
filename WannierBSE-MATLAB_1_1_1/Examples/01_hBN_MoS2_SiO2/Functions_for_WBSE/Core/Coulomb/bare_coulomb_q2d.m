function Vq = bare_coulomb_q2d(q, d)
%==========================================================================
% BARE_COULOMB_Q2D
%
% DESCRIPTION:
%   Evaluates the strict-2D or finite-thickness bare Coulomb kernel.
%   The q=0 value is a quadrature placeholder; the physical self term is
%   supplied by the local-cell singular integral.
%
% INPUT ARGUMENTS:
%   q : momentum magnitudes.
%   d : effective material thickness.
%
% OUTPUT:
%   Vq : bare quasi-2D Coulomb kernel evaluated at q.
%==========================================================================

Vq = zeros(size(q));
positive_q = (q > 0);
if d == 0
    Vq(positive_q) = 2 * pi ./ q(positive_q);
else
    q_positive = q(positive_q);
    Vq(positive_q) = (1 - ((1 - exp(-(d .* q_positive))) ./ (d .* q_positive))) .* ...
        ((4 * pi) ./ (d .* (q_positive .^ 2)));
end
end
