%==========================================================================
% WF_TO_MAT
%
% DESCRIPTION:
%   Orchestrates Wannier-function preprocessing by importing spinor XSF data, assembling complex components, shifting grids, normalizing wavefunctions, and exporting WBSE-ready caches.
%
% INPUT ARGUMENTS:
%   None. Paths and optional controls are loaded from the active project configuration.
%
% OUTPUT:
%   None. Legacy WBSE Wannier caches and preprocessing diagnostics are written.
%==========================================================================

fprintf('[PRE  ] Wannier: Preparing real-space Wannier functions...\n');
t_wan_total = tic;
% OPT-INSTRUMENTATION
t_opt_pre_total = tic;

% 1. Initialize Paths
% OPT-INSTRUMENTATION
t_opt_setup = tic;
Paths = init_wannier_paths();

% 2. Read User Parameters from exchange_control.txt
if evalin('base', 'exist(''ParamPath'', ''var'')')
    ParamPath = evalin('base', 'ParamPath');
    param_file = fullfile(ParamPath, 'exchange_control.txt');
else
    param_file = fullfile(Paths.parameters, 'exchange_control.txt');
end
user_params = read_params(param_file);

% Extract the cleanup flag (default to true if not found)
if isfield(user_params, 'cleanup_xsf')
    cleanup_xsf = parse_cleanup_xsf(user_params.cleanup_xsf);
else
    cleanup_xsf = false; 
end
% Advanced/developer-only WF diagnostic; absent means disabled.
if isfield(user_params, 'Validate_WF_Overlap')
    Validate_WF_Overlap = parse_optional_boolean(user_params.Validate_WF_Overlap, ...
                                                 'Validate_WF_Overlap', param_file);
else
    Validate_WF_Overlap = false;
end

% 4. Create the Workspace
if ~exist(Paths.workspace, 'dir')
    mkdir(Paths.workspace);
end
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_setup, 'cache_check', 'exclusive', '1', 'NA');

% 5. Execution Pipeline
% (The scripts below will now have access to 'Paths' and 'cleanup_xsf')
fprintf('  -> Importing XSF volumetric data...\n');
t_step = tic;
% OPT-INSTRUMENTATION
t_opt_xsf = tic;
run('test_up_real.m');
run('test_up_imag.m');
run('test_down_real.m');
run('test_down_imag.m');
fprintf('  -> XSF volumetric data imported in %.2f s\n', toc(t_step));
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_xsf, 'xsf_values_read', 'inclusive', '88', '4199040');

% OPT-INSTRUMENTATION
t_opt_cleanup = tic;
run('Delete_xsf.m');
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_cleanup, 'cleanup', 'exclusive', '1', 'NA');

t_step = tic;
% OPT-INSTRUMENTATION
t_opt_spinor = tic;
run('Step2_SaveWF_spinor_v2.m');
fprintf('  -> Spinor basis assembled in %.2f s\n', toc(t_step));
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_spinor, 'spinor_assembly', 'inclusive', '44', '4199040x22');

t_step = tic;
% OPT-INSTRUMENTATION
t_opt_shift = tic;
run('Step3_shift_r_and_z.m');
fprintf('  -> Spatial grid shifted in %.2f s\n', toc(t_step));
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_shift, 'shift_apply', 'inclusive', '22', '4199040x22');

t_step = tic;
% OPT-INSTRUMENTATION
t_opt_norm = tic;
run('Step4_Normalize_WFs.m');
fprintf('  -> Wavefunctions normalized in %.2f s\n', toc(t_step));
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_norm, 'normalization_integral', 'inclusive', '22', '4199040x22');

t_step = tic;
% OPT-INSTRUMENTATION
t_opt_reorder = tic;
run('Step5_reorder_WFs_TMD_ML_only.m');
fprintf('  -> Wannier basis reordered for the active material convention in %.2f s\n', toc(t_step));
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_reorder, 'basis_reorder', 'inclusive', '22', '4199040x22');

fprintf('[DONE ] Wannier: Preprocessing completed in %.2f s. Cleanup flag: %d\n', toc(t_wan_total), cleanup_xsf);
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_pre_total, 'preprocessing_total', 'inclusive', '1', 'NA');
