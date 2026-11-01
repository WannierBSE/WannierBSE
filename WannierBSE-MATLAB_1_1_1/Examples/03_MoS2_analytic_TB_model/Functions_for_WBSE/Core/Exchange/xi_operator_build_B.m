function [M_small,M_large] = xi_operator_build_B(hf,hf0,z_point_t,G,kex,b1)
%==========================================================================
% XI_OPERATOR_BUILD_B
%
% DESCRIPTION:
%   Builds reusable rho2 B operators for one reciprocal vector. The
%   operators are constructed once for the fixed G, kex, and z grid and
%   reused for all R1/R2 blocks.
%
% INPUT ARGUMENTS:
%   hf, hf0 : spline integration function handles.
%   z_point_t : increasing z grid.
%   G, kex, b1 : reciprocal vector, exchange momentum, and lattice vector.
%
% OUTPUT:
%   M_small, M_large : complex Nz-by-Nz integration operators.
%
% NOTES:
%   No conjugation or real cast is applied to the operators.
%==========================================================================

z_point_t = z_point_t(:);
Nz = numel(z_point_t);
if Nz < 2 || any(diff(z_point_t) <= 0)
    error('xi_operator_build_B:InvalidGrid','z_point_t must be increasing.');
end
if ~isequal(size(G),[1,2]) || ~isequal(size(kex),[1,3])
    error('xi_operator_build_B:InvalidMomentumShape','G must be 1-by-2 and kex 1-by-3.');
end
hz = z_point_t(2) - z_point_t(1);
M_small = complex(zeros(Nz,Nz),zeros(Nz,Nz));
M_large = complex(zeros(Nz,Nz),zeros(Nz,Nz));
for k = 1:Nz
    e = complex(zeros(Nz,1),zeros(Nz,1));
    e(k) = 1;
    [M_small(:,k),M_large(:,k)] = local_B(e,G,kex,z_point_t,b1,hf,hf0,hz);
end
end

function [B_small,B_large] = local_B(y,G,kex,z_point_t,b1,hf,hf0,hz)
Nz = numel(y);
B_small = complex(zeros(Nz,1),zeros(Nz,1));
B_large = complex(zeros(Nz,1),zeros(Nz,1));
c = spline(z_point_t,y).coefs;
zero_tol = 1e-7;
q = norm(G(1:2) + kex(1:2));

if (norm(G) < zero_tol) && (norm(kex) > zero_tol) && (norm(kex) < norm(b1))
    for zi = 1:Nz
        B_small(zi) = sum(hf0(c(1:zi-1,1),c(1:zi-1,2),c(1:zi-1,3), ...
            c(1:zi-1,4),0,z_point_t(1:zi-1),hz));
        B_large(zi) = sum(hf0(c(zi:end,1),c(zi:end,2),c(zi:end,3), ...
            c(zi:end,4),0,z_point_t(zi:end-1),hz));
    end
elseif (norm(G) < zero_tol) && (norm(kex) < zero_tol)
    for zi = 1:Nz-1
        B_small(zi+1) = sum(hf0(c(1:zi,1),c(1:zi,2),c(1:zi,3), ...
            c(1:zi,4),-q,z_point_t(1:zi),hz));
        B_large(zi) = sum(hf0(c(zi:end,1),c(zi:end,2),c(zi:end,3), ...
            c(zi:end,4),q,z_point_t(zi:end-1),hz));
    end
else
    for zi = 1:Nz-1
        B_small(zi+1) = sum(hf(c(1:zi,1),c(1:zi,2),c(1:zi,3), ...
            c(1:zi,4),-q,z_point_t(1:zi),hz,z_point_t(zi+1)));
        B_large(zi) = sum(hf(c(zi:end,1),c(zi:end,2),c(zi:end,3), ...
            c(zi:end,4),q,z_point_t(zi:end-1),hz,z_point_t(zi)));
    end
end
end
