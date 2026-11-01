function Paths = init_wannier_paths()
%==========================================================================
% INIT_WANNIER_PATHS
%
% DESCRIPTION:
%   Initializes the standardized directory layout used by the Wannier-function preprocessing scripts.
%
% INPUT ARGUMENTS:
%   None. The project root is inferred from this file's location.
%
% OUTPUT:
%   Paths : structure of source, workspace, temporary, parameter, and output directories.
%==========================================================================

    engine_dir = fileparts(mfilename('fullpath'));
    
    % Navigate two levels up to the Project Root
    project_root = fullfile(engine_dir, '..', '..');

    % --- Input Data Directories ---
    % Path to raw user inputs (e.g., wannier90.win, wannier90_hr.dat)
    Paths.user_input = fullfile(project_root, 'User_input');
    
    % Path to the volumetric .xsf component folders
    Paths.xsf_root = fullfile(Paths.user_input, 'Wannier_functions_xsf');

    % --- Control & Library Directories ---
    % Path to control files (e.g., exchange_control.txt)
    Paths.parameters = fullfile(project_root, 'Parameters');
    
    % Path to the shared library containing read_structure.m and read_params.m
    Paths.shared_functions = fullfile(engine_dir, '..', 'Loaders');

    % --- Precomputed Data & Workspace ---
    % Root path for precomputed data
    Paths.precomputed  = fullfile(project_root, 'Precomputed_data');

    % Destination for finalized, high-precision Wannier function data
    Paths.final_output = fullfile(Paths.precomputed, 'Wannier_functions_WBSE');

    % Legacy-only workspace nested inside the self-contained cache.
    Paths.workspace    = fullfile(Paths.final_output, 'Wannier_workspace');

    % --- Temporary Processing Containers (Inside Workspace) ---
    Paths.temp_output     = fullfile(Paths.workspace, 'Output');
    Paths.temp_raw        = fullfile(Paths.workspace, 'WF_raw');
    Paths.temp_shifted    = fullfile(Paths.workspace, 'WF_raw_shifted');
    Paths.temp_normalized = fullfile(Paths.workspace, 'Normalized_WFs');

end
