function [W_up_norm, W_down_norm] = Normalize_WF(W_up, W_down, Na1, Na2, Na3, J, hu, hv, hz, u_point, v_point, z_point)
%==========================================================================
% NORMALIZE_WF
%
% DESCRIPTION:
%   Normalizes spinor Wannier functions by spline integration of the full three-dimensional probability
%   density.
%
% INPUT ARGUMENTS:
%   W_up, W_down : raw spinor components.
%   Na1, Na2, Na3 : grid dimensions.
%   J, hu, hv, hz : integration metric and spacings.
%   u_point, v_point, z_point : spline grid coordinates.
%
% OUTPUT:
%   W_up_norm, W_down_norm : normalized spinor Wannier components.
%==========================================================================

[~, hf0] = SPint_uv; % Integration weights from spline utility
nB = size(W_up, 2);  % Number of Wannier functions
Nu = Na1; Nv = Na2; Nz = Na3;

W_up_norm = zeros(size(W_up));
W_down_norm = zeros(size(W_down));
N = zeros(nB, 1);

% --- Normalization Loop ---
for i = 1:nB
    % OPT-INSTRUMENTATION
    t_opt_integral = tic;
    % Calculate the probability density: |W_up|^2 + |W_down|^2
    Wij = (conj(W_up(:,i)).*W_up(:,i)) + (conj(W_down(:,i)).*W_down(:,i));
    
    % Reshape and permute for spline processing [v, z, u]
    RW = reshape(Wij, [Nu, Nv, Nz]);
    RW = permute(RW, [2, 3, 1]);
    
    % 1. Integration along the u-direction
    SPu = spline(u_point, RW);
    c1 = reshape(SPu.coefs(:,1), [Nv, Nz, Nu-1]); cu1 = permute(c1, [3, 1, 2]);
    c2 = reshape(SPu.coefs(:,2), [Nv, Nz, Nu-1]); cu2 = permute(c2, [3, 1, 2]);
    c3 = reshape(SPu.coefs(:,3), [Nv, Nz, Nu-1]); cu3 = permute(c3, [3, 1, 2]);
    c4 = reshape(SPu.coefs(:,4), [Nv, Nz, Nu-1]); cu4 = permute(c4, [3, 1, 2]);
    
    Su = J * sum(hf0(cu1, cu2, cu3, cu4, u_point(1:end-1), hu), 1);
    
    % 2. Integration along the v-direction
    SPv = spline(v_point, transpose(squeeze(Su)));
    cv1 = transpose(reshape(SPv.coefs(:,1), [Nz, Nv-1]));
    cv2 = transpose(reshape(SPv.coefs(:,2), [Nz, Nv-1]));
    cv3 = transpose(reshape(SPv.coefs(:,3), [Nz, Nv-1]));
    cv4 = transpose(reshape(SPv.coefs(:,4), [Nz, Nv-1]));
    
    Sv = sum(hf0(cv1, cv2, cv3, cv4, v_point(1:end-1), hv), 1);
    
    % 3. Integration along the z-direction
    SPz = spline(z_point, Sv);
    cz1 = SPz.coefs(:,1);
    cz2 = SPz.coefs(:,2);
    cz3 = SPz.coefs(:,3);
    cz4 = SPz.coefs(:,4);
    
    Sz = sum(hf0(cz1, cz2, cz3, cz4, z_point(1:end-1), hz));
    
    % OPT-INSTRUMENTATION
    opt_wf_timing('stop', t_opt_integral, 'normalization_integral', 'exclusive', num2str(i), sprintf('%dx%dx%d', Nu, Nv, Nz));
    % --- Apply Normalization ---
    % OPT-INSTRUMENTATION
    t_opt_scale = tic;
    norm_const = sqrt(Sz); 
    N(i) = norm_const; 
    W_up_norm(:,i)   = W_up(:,i) ./ norm_const;
    W_down_norm(:,i) = W_down(:,i) ./ norm_const;
    % OPT-INSTRUMENTATION
    opt_wf_timing('stop', t_opt_scale, 'normalization_scale', 'exclusive', num2str(i), sprintf('%dx%d', size(W_up,1), 1));
end

end
