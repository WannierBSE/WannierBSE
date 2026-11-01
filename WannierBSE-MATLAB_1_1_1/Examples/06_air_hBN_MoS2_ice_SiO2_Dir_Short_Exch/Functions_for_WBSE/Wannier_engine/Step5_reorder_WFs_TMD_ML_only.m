%==========================================================================
% STEP5_REORDER_WFS_TMD_ML_ONLY
%
% DESCRIPTION:
%   Reorders normalized TMD monolayer Wannier functions into the WBSE spin-block basis and exports final metadata.
%
% INPUT ARGUMENTS:
%   None. Normalized Wannier data and paths are obtained from the workspace.
%
% OUTPUT:
%   None. Reordered Wannier arrays and final metadata are written to the workspace.
%==========================================================================

order1 = transpose(1:22);  % W90 output
order2 = [1;12;2;13;3;14;4;15;5;16;6;17;7;18;8;19;9;20;10;21;11;22]; % our convention

%----------read Wannier functions---------
% Load the final normalized data from the workspace
load(fullfile(Paths.temp_normalized, 'W_up.mat'))
load(fullfile(Paths.temp_normalized, 'W_down.mat'))
load(fullfile(Paths.temp_normalized, 'r.mat'))

W_up_origin = W_up;
W_down_origin = W_down;
clear W_up W_down

% Perform Reordering
% OPT-INSTRUMENTATION
t_opt_reorder = tic;
W_up = zeros(size(W_up_origin));
W_down = zeros(size(W_down_origin));
for n=1:size(W_up,2)
    W_up(:,order2(n)) = W_up_origin(:,order1(n));
    W_down(:,order2(n)) = W_down_origin(:,order1(n));
end
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_reorder, 'basis_reorder', 'exclusive', num2str(size(W_up,2)), sprintf('%dx%d', size(W_up,1), size(W_up,2)));

% Create final destination folder if it doesn't exist
if ~exist(Paths.final_output, 'dir')
    mkdir(Paths.final_output);
end

% Save results to the final permanent directory (Precomputed_data/Wannier_functions_WBSE/)
save(fullfile(Paths.final_output, 'WF_up.mat'),'W_up','-v7.3')
save(fullfile(Paths.final_output, 'WF_down.mat'),'W_down','-v7.3')
save(fullfile(Paths.final_output, 'r.mat'),'r','-v7.3')

% Copy metadata files to final destination
copyfile(fullfile(Paths.temp_normalized, 'GridInfo.txt'), fullfile(Paths.final_output, 'GridInfo.txt'))
copyfile(fullfile(Paths.temp_normalized, 'WF_AtomPosition.txt'), fullfile(Paths.final_output, 'WF_AtomPosition.txt'))

% --- Reorder and Save WF_centers.txt ---
% OPT-INSTRUMENTATION
t_opt_centers = tic;
wf_centers_in = fullfile(Paths.parameters, 'WF_centers.txt');
wf_centers_out = fullfile(Paths.final_output, 'WF_centers_WBSE.txt');

reorder_WF_centers(wf_centers_in, wf_centers_out);
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_centers, 'wf_center_processing', 'exclusive', num2str(size(W_up,2)), 'NA');

%-------delete data in previous step (Normalized_WFs)--------
if exist(fullfile(Paths.temp_normalized, 'W_up.mat'), 'file'), delete(fullfile(Paths.temp_normalized, 'W_up.mat')); end
if exist(fullfile(Paths.temp_normalized, 'W_down.mat'), 'file'), delete(fullfile(Paths.temp_normalized, 'W_down.mat')); end
if exist(fullfile(Paths.temp_normalized, 'r.mat'), 'file'), delete(fullfile(Paths.temp_normalized, 'r.mat')); end

% Final Workspace Cleanup: Remove the entire temporary directory
if exist(Paths.workspace, 'dir')
    rmdir(Paths.workspace, 's');
end
