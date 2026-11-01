function [Xi] = xi_integral( ...
    hf0, hf, rho1, rho2, G, kex, z_point_t, NB, b1, jj_index, j_index_tot, ll_index, l_index_tot)
%==========================================================================
% XI_INTEGRAL
%
% DESCRIPTION:
%   Contracts z-resolved pair densities through analytic spline kernels to form Xi exchange integrals.
%
% INPUT ARGUMENTS:
%   hf0, hf : spline integration function handles.
%   rho1, rho2 : z-resolved pair-density arrays.
%   G, kex, z_point_t : reciprocal vector, exchange momentum, and z grid.
%   NB, b1 : basis-function count and reciprocal lattice vector.
%   jj_index, j_index_tot, ll_index, l_index_tot : Wannier-index neighborhoods.
%
% OUTPUT:
%   Xi : Wannier-basis Xi tensor for one G vector and R pair.
%==========================================================================

    hz = z_point_t(2) - z_point_t(1);
    zero_tol = 1e-7;
    Xi = zeros(NB, NB, NB, NB);
    
    % Momentum magnitudes
    NGkex = norm(G(1:2) + kex(1:2));
    
    % --- Step 1: Integral over z2 (Green's Function Splitting) ---
    for ll = ll_index
        l_range = l_index_tot{ll};
        for l = l_range
            B_small = zeros(size(rho2, 3), 1);
            B_large = zeros(size(rho2, 3), 1);
            
            % Spline the density rho(z) once for this pair
            BSp = spline(z_point_t, squeeze(rho2(ll, l, :)));
            c = BSp.coefs; % Order: c3, c2, c1, c0
            
            if (norm(G) < zero_tol) && (norm(kex) > zero_tol) && (norm(kex) < norm(b1))
                % Special momentum case for shifted integration
                for zi = 1:size(rho2, 3)
                    % Small z region (analytical polynomial only)
                    % Note: Logic replicates WH's handle usage for stability
                    B_small(zi, 1) = sum(hf0(c(1:zi-1, 1), c(1:zi-1, 2), c(1:zi-1, 3), c(1:zi-1, 4), ...
                                             0, z_point_t(1:zi-1), hz));
                    
                    % Large z region
                    B_large(zi, 1) = sum(hf0(c(zi:end, 1), c(zi:end, 2), c(zi:end, 3), c(zi:end, 4), ...
                                             0, z_point_t(zi:end-1), hz));
                end
                
            elseif (norm(G) < zero_tol) && (norm(kex) < zero_tol)
                % Strictly Long Range / local case
                for zi = 1:size(rho2, 3) - 1
                    B_small(zi+1, 1) = sum(hf0(c(1:zi, 1), c(1:zi, 2), c(1:zi, 3), c(1:zi, 4), ...
                                               -NGkex, z_point_t(1:zi), hz));
                    B_large(zi, 1) = sum(hf0(c(zi:end, 1), c(zi:end, 2), c(zi:end, 3), c(zi:end, 4), ...
                                             NGkex, z_point_t(zi:end-1), hz));
                end
                
            else
                % General Case: G ~= 0 or large kex (Uses full analytical kernel)
                for zi = 1:size(rho2, 3) - 1
                    % Integration over [0, z1]
                    B_small(zi+1, 1) = sum(hf(c(1:zi, 1), c(1:zi, 2), c(1:zi, 3), c(1:zi, 4), ...
                                              -NGkex, z_point_t(1:zi), hz, z_point_t(zi+1)));
                    
                    % Integration over [z1, d]
                    B_large(zi, 1) = sum(hf(c(zi:end, 1), c(zi:end, 2), c(zi:end, 3), c(zi:end, 4), ...
                                             NGkex, z_point_t(zi:end-1), hz, z_point_t(zi)));
                end
            end
            
            % --- Step 2: Integral over z1 ---
            for jj = jj_index
                j_range = j_index_tot{jj};
                for j = j_range
                    % Spline the product of density and z2-integral
                    B2_small = spline(z_point_t, squeeze(conj(rho1(jj, j, :))) .* B_small);
                    cs = B2_small.coefs;
                    S_small = sum(hf0(cs(:, 1), cs(:, 2), cs(:, 3), cs(:, 4), 0, z_point_t(1:end-1), hz));
                    
                    B2_large = spline(z_point_t, squeeze(conj(rho1(jj, j, :))) .* B_large);
                    cl = B2_large.coefs;
                    S_large = sum(hf0(cl(:, 1), cl(:, 2), cl(:, 3), cl(:, 4), 0, z_point_t(1:end-1), hz));
                    
                    Xi(jj, j, ll, l) = S_small + S_large;
                end
            end
        end
    end
end
