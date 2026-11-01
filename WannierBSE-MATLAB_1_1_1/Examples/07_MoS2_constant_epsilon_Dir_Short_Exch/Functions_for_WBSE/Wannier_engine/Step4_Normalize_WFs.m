%==========================================================================
% STEP4_NORMALIZE_WFS
%
% DESCRIPTION:
%   Normalizes spinor Wannier functions and records the overlap matrix used to validate orthonormality.
%
% INPUT ARGUMENTS:
%   None. Shifted Wannier data are loaded from the active preprocessing workspace.
%
% OUTPUT:
%   None. Normalized Wannier arrays and diagnostics are written to the workspace.
%==========================================================================

[a1,a2,a3,a,tau,N_atoms] = read_structure;
%----------read Wannier functions---------
% Load data from the shifted workspace folder
load(fullfile(Paths.temp_shifted, 'W_up.mat'))
load(fullfile(Paths.temp_shifted, 'W_down.mat'))
load(fullfile(Paths.temp_shifted, 'r.mat'))

GridInfo = importdata(fullfile(Paths.temp_shifted, 'GridInfo.txt'));
Atoms = importdata(fullfile(Paths.temp_shifted, 'WF_AtomPosition.txt'));

Na1 = GridInfo(1,1);
Na2 = GridInfo(1,2);
Na3 = GridInfo(1,3);
A1s = GridInfo(3,:);
A2s = GridInfo(4,:);
A3s = GridInfo(5,:);
%----------read Wannier functions---------

%------------normalization-------------
% Function calls for CoordinateTransformation and Normalize_WF 
% rely on the Wannier_engine folder being the current directory.
grid_Nu = Na1;
grid_Nv = Na2;
grid_Nz = Na3;
% OPT-INSTRUMENTATION
t_opt_geometry = tic;
[J,hu,hv,hz,u,v,u_point,v_point,z_point] = CoordinateTransformation(r,a1,a2,grid_Nz);
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_geometry, 'normalization_geometry', 'exclusive', '1', sprintf('%dx%dx%d', Na1, Na2, Na3));

% OPT-INSTRUMENTATION
t_opt_norm_call = tic;
[W_up_norm,W_down_norm] = Normalize_WF(W_up,W_down,Na1,Na2,Na3,J,hu,hv,hz,u_point,v_point,z_point);
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_norm_call, 'normalization_integral', 'inclusive', num2str(size(W_up,2)), sprintf('%dx%d', size(W_up,1), size(W_up,2)));
%------------normalization-------------

clear W_up W_down
W_up = W_up_norm;
W_down = W_down_norm;
clear W_up_norm W_down_norm

%---------------inner product--------------
if Validate_WF_Overlap
    % OPT-INSTRUMENTATION
    t_opt_overlap = tic;
    [~,hf0] = SPint_uv;
    M = zeros(size(W_up,2),size(W_up,2));
    for i=1:size(W_up,2)
        for j=1:size(W_up,2)
            Wij = (conj(W_up(:,i)).*W_up(:,j))+(conj(W_down(:,i)).*W_down(:,j));

            RW = reshape(Wij,[grid_Nu,grid_Nv,grid_Nz]);
            RW = permute(RW,[2,3,1]);
            SPu = spline(u_point,RW);
            c1 = SPu.coefs(:,1);
            c2 = SPu.coefs(:,2);
            c3 = SPu.coefs(:,3);
            c4 = SPu.coefs(:,4);
            c1 = reshape(c1,[grid_Nv,grid_Nz,grid_Nu-1]);
            cu1 = permute(c1,[3,1,2]);
            c2 = reshape(c2,[grid_Nv,grid_Nz,grid_Nu-1]);
            cu2 = permute(c2,[3,1,2]);
            c3 = reshape(c3,[grid_Nv,grid_Nz,grid_Nu-1]);
            cu3 = permute(c3,[3,1,2]);
            c4 = reshape(c4,[grid_Nv,grid_Nz,grid_Nu-1]);
            cu4 = permute(c4,[3,1,2]);

            % integration u
            Su=J*sum(hf0(cu1,cu2,cu3,cu4,u_point(1:end-1),hu),1);

            % integration v
            SPv = spline(v_point,transpose(squeeze(Su)));
            cv1 = SPv.coefs(:,1);
            cv2 = SPv.coefs(:,2);
            cv3 = SPv.coefs(:,3);
            cv4 = SPv.coefs(:,4);
            cv1 = transpose(reshape(cv1,[grid_Nz,grid_Nv-1]));
            cv2 = transpose(reshape(cv2,[grid_Nz,grid_Nv-1]));
            cv3 = transpose(reshape(cv3,[grid_Nz,grid_Nv-1]));
            cv4 = transpose(reshape(cv4,[grid_Nz,grid_Nv-1]));
            Sv = sum(hf0(cv1,cv2,cv3,cv4,v_point(1:end-1),hv),1);

            % integration z
            SPz = spline(z_point,Sv);
            cz1 = SPz.coefs(:,1);
            cz2 = SPz.coefs(:,2);
            cz3 = SPz.coefs(:,3);
            cz4 = SPz.coefs(:,4);
            Sz = sum(hf0(cz1,cz2,cz3,cz4,z_point(1:end-1),hz));
            M(i,j) = Sz;
            M(i,j) = conj(Sz);
        end
    end
    % OPT-INSTRUMENTATION
    opt_wf_timing('stop', t_opt_overlap, 'overlap_validation', 'exclusive', num2str(size(W_up,2)^2), sprintf('%dx%d', size(W_up,2), size(W_up,2)));

    diag_deviation = max(abs(diag(M) - 1));
    off_diagonal = M;
    off_diagonal(1:size(M,1)+1:end) = 0;
    max_off_diagonal = max(abs(off_diagonal(:)));
    hermiticity_residual = norm(M - M', 'fro') / max(norm(M, 'fro'), eps);
    nonfinite_count = sum(isnan(M(:)) | isinf(M(:)));
    fprintf('  -> WF overlap diagnostic: max diagonal deviation %.6e, max off-diagonal %.6e, Hermiticity residual %.6e, nonfinite entries %d\n', ...
            diag_deviation, max_off_diagonal, hermiticity_residual, nonfinite_count);
    if nonfinite_count > 0
        warning('WBSE:NonfiniteWFOverlap', ...
                'WF overlap diagnostic contains %d NaN/Inf entries.', nonfinite_count);
    end

    % Save the overlap matrix M in the main workspace root for logging.
    % M.mat is intentionally not removed when validation is disabled.
    % OPT-INSTRUMENTATION
    t_opt_m_io = tic;
    save(fullfile(Paths.workspace, 'M.mat'),'M','-v7.3')
    % OPT-INSTRUMENTATION
    opt_wf_timing('stop', t_opt_m_io, 'cache_write', 'exclusive', '1', '22x22');
else
    % OPT-INSTRUMENTATION
    fprintf('[OPT ] WF implementation=MATLAB phase=overlap_validation timing_scope=exclusive time_s=0 items=0 input_shape=22x22 status=skipped\n');
end
%---------------inner product--------------

% Create workspace/Normalized_WFs if it doesn't exist
if ~exist(Paths.temp_normalized, 'dir')
    mkdir(Paths.temp_normalized);
end

% Save normalized WFs to workspace/Normalized_WFs
% OPT-INSTRUMENTATION
t_opt_io = tic;
save(fullfile(Paths.temp_normalized, 'W_up.mat'),'W_up','-v7.3')
save(fullfile(Paths.temp_normalized, 'W_down.mat'),'W_down','-v7.3')
save(fullfile(Paths.temp_normalized, 'r.mat'),'r','-v7.3')

% Copy metadata files from the previous step
copyfile(fullfile(Paths.temp_shifted, 'GridInfo.txt'), fullfile(Paths.temp_normalized, 'GridInfo.txt'))
copyfile(fullfile(Paths.temp_shifted, 'WF_AtomPosition.txt'), fullfile(Paths.temp_normalized, 'WF_AtomPosition.txt'))
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_io, 'cache_write', 'exclusive', '3', 'NA')

%-------delete data in previous step (WF_raw_shifted)--------
if exist(fullfile(Paths.temp_shifted, 'W_up.mat'), 'file'), delete(fullfile(Paths.temp_shifted, 'W_up.mat')); end
if exist(fullfile(Paths.temp_shifted, 'W_down.mat'), 'file'), delete(fullfile(Paths.temp_shifted, 'W_down.mat')); end
if exist(fullfile(Paths.temp_shifted, 'r.mat'), 'file'), delete(fullfile(Paths.temp_shifted, 'r.mat')); end
%-------delete data in previous step--------
