function epslnq = epsilon_Q2D(q_data, epsQ2D_data, q)
%==========================================================================
% EPSILON_Q2D
%
% DESCRIPTION:
%   Interpolates the tabulated quasi-2D dielectric function at requested momentum magnitudes.
%
% INPUT ARGUMENTS:
%   q_data : tabulated q-grid.
%   epsQ2D_data : dielectric values on q_data.
%   q : scalar or array of query momentum magnitudes.
%
% OUTPUT:
%   epslnq : interpolated dielectric values with the same shape as q.
%==========================================================================

    epslnq = zeros(size(q)); % preallocate to match q's shape

    % Below three logical masks cover all cases for each q element
    t_smaller = q < min(q_data);
    t_middle  = q >= min(q_data) & q < max(q_data(end-1));
    t_larger  = q >= max(q_data(end-1));

    epslnq(t_smaller) = 1;
    epslnq(t_middle)  = interp1(q_data, epsQ2D_data, q(t_middle), 'spline');
    epslnq(t_larger)  = interp1(q_data, epsQ2D_data, q(t_larger), 'linear');
end
