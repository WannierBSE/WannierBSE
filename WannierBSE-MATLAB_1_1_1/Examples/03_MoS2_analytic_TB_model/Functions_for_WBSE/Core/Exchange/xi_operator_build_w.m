function w = xi_operator_build_w(hf0,z_point_t)
%==========================================================================
% XI_OPERATOR_BUILD_W
%
% DESCRIPTION:
%   Builds the reusable fixed-grid rho1 integration operator. Each response
%   is generated from a complex unit basis through MATLAB spline and the
%   production hf0 path.
%
% INPUT ARGUMENTS:
%   hf0 : non-oscillatory spline integration function handle.
%   z_point_t : increasing z grid.
%
% OUTPUT:
%   w : complex Nz-by-1 fixed-grid integration operator.
%==========================================================================

z_point_t = z_point_t(:);
Nz = numel(z_point_t);
if Nz < 2 || any(diff(z_point_t) <= 0)
    error('xi_operator_build_w:InvalidGrid','z_point_t must be increasing.');
end
hz = z_point_t(2) - z_point_t(1);
w = complex(zeros(Nz,1),zeros(Nz,1));
for k = 1:Nz
    e = complex(zeros(Nz,1),zeros(Nz,1));
    e(k) = 1;
    S = spline(z_point_t,e);
    w(k) = sum(hf0(S.coefs(:,1),S.coefs(:,2),S.coefs(:,3), ...
        S.coefs(:,4),0,z_point_t(1:end-1),hz));
end
end
