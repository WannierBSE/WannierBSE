function pool = ensure_parallel_pool(requestedWorkers, stageName, protectExistingPool)
%==========================================================================
% ENSURE_PARALLEL_POOL
%
% DESCRIPTION:
%   Returns an exactly sized and verified local MATLAB process pool for a
%   parallel stage. A request for one worker is treated as serial execution;
%   no pool is created or resized in that case. Existing pools can optionally
%   be protected from resizing or deletion.
%
% INPUT ARGUMENTS:
%   requestedWorkers : positive integer number of process workers.
%   stageName : optional label used in diagnostics and validation messages.
%   protectExistingPool : optional logical flag preserving a user-owned pool.
%
% OUTPUT:
%   pool : current or newly created local process-pool object; for a serial
%          request, this may be empty when no pool existed.
%==========================================================================

if nargin < 2 || isempty(stageName)
    stageName = 'parallel';
end
if nargin < 3 || isempty(protectExistingPool)
    protectExistingPool = false;
end
if ~(isnumeric(requestedWorkers) && isscalar(requestedWorkers) && ...
        isfinite(requestedWorkers) && requestedWorkers >= 1 && ...
        requestedWorkers == floor(requestedWorkers))
    error('WBSE:InvalidParallelWorkerCount', ...
          'requestedWorkers must be a positive finite integer.');
end

pool = gcp('nocreate');
if requestedWorkers == 1
    fprintf('[OPT  ] Parallel: stage=%s serial execution requested; no pool created or resized.\n', stageName);
    return;
end
action = 'reused';
if isempty(pool)
    action = 'created';
    fprintf('[OPT  ] Parallel: stage=%s pool action=created requested_workers=%d\n', stageName, requestedWorkers);
    tPool = tic;
    pool = parpool('local', requestedWorkers);
    fprintf('[OPT  ] Parallel: stage=%s pool_startup_s=%.6f\n', stageName, toc(tPool));
else
    if pool.NumWorkers ~= requestedWorkers
        if protectExistingPool
            fprintf('[OPT  ] Parallel: stage=%s reusing user-owned pool workers=%d requested_workers=%d; pool unchanged.\n', ...
                    stageName, pool.NumWorkers, requestedWorkers);
        else
            fprintf('[OPT  ] Parallel: stage=%s pool action=resized old_workers=%d requested_workers=%d\n', ...
                    stageName, pool.NumWorkers, requestedWorkers);
            delete(pool);
            action = 'resized';
            tPool = tic;
            pool = parpool('local', requestedWorkers);
            fprintf('[OPT  ] Parallel: stage=%s pool_startup_s=%.6f\n', stageName, toc(tPool));
        end
    else
        fprintf('[OPT  ] Parallel: stage=%s pool action=reused workers=%d\n', stageName, pool.NumWorkers);
    end
end

try
    verify_parallel_runtime(pool, requestedWorkers, stageName);
catch ME
    if strcmp(action, 'reused') && ~protectExistingPool
        fprintf('[OPT  ] Parallel: stage=%s recreating pool because existing worker policy was not verified.\n', stageName);
        delete(pool);
        tPool = tic;
        pool = parpool('local', requestedWorkers);
        fprintf('[OPT  ] Parallel: stage=%s pool_startup_s=%.6f\n', stageName, toc(tPool));
        verify_parallel_runtime(pool, requestedWorkers, stageName);
    else
        rethrow(ME);
    end
end

if ~protectExistingPool && pool.NumWorkers ~= requestedWorkers
    error('WBSE:ParallelWorkerCountMismatch', ...
          'Stage %s requested %d workers but MATLAB provided %d.', ...
          stageName, requestedWorkers, pool.NumWorkers);
end
if protectExistingPool && pool.NumWorkers ~= requestedWorkers
    fprintf('[WARN ] Parallel: stage=%s using user-owned pool with %d workers instead of requested %d.\n', ...
            stageName, pool.NumWorkers, requestedWorkers);
end
attach_hbse_source_to_pool(pool);
end

function attach_hbse_source_to_pool(pool)
% Attach the HBSE source explicitly because process workers do not reliably
% resolve the workstation's /G project path during parfor source transfer.
utils_dir = fileparts(mfilename('fullpath'));
functions_root = fileparts(utils_dir);
project_root = fileparts(functions_root);
worker_project_root = strrep(project_root, '/G/OscarJGomezS', '/home/oscar');
worker_functions_root = fullfile(worker_project_root, 'Functions_for_WBSE');
worker_hbse_source = fullfile(worker_functions_root, 'Core', 'build_HBSE_flattened.m');
client_hbse_source = fullfile(project_root, 'Functions_for_WBSE', 'Core', 'build_HBSE_flattened.m');
worker_direct_source = fullfile(worker_functions_root, 'Core', 'Direct', 'direct_coulomb_source_group.m');
client_direct_source = fullfile(project_root, 'Functions_for_WBSE', 'Core', 'Direct', 'direct_coulomb_source_group.m');
if ~isfile(client_hbse_source) || ~isfile(worker_hbse_source) || ...
        ~isfile(client_direct_source) || ~isfile(worker_direct_source)
    error('WBSE:HBSESourceMissing', ...
        'Cannot attach HBSE/direct source. HBSE client=%s worker=%s; direct client=%s worker=%s', ...
        client_hbse_source, worker_hbse_source, client_direct_source, worker_direct_source);
end
required_sources = matlab.codetools.requiredFilesAndProducts(client_hbse_source);
attached_sources = unique([{client_hbse_source, worker_hbse_source, ...
    client_direct_source, worker_direct_source}, required_sources(:).']);
addAttachedFiles(pool, attached_sources);
% Do not use genpath(project_root) here: it exposes .codex_backups and
% inspection snapshots to workers, where an older direct kernel can shadow
% the production G_dir_max-aware implementation.
worker_paths = {worker_functions_root, ...
    fullfile(worker_functions_root, 'Loaders'), ...
    fullfile(worker_functions_root, 'Utils'), ...
    fullfile(worker_functions_root, 'TB'), ...
    fullfile(worker_functions_root, 'Kmesh'), ...
    fullfile(worker_functions_root, 'Dielectric'), ...
    fullfile(worker_functions_root, 'Wannier_engine'), ...
    fullfile(worker_functions_root, 'Core'), ...
    fullfile(worker_functions_root, 'Core', 'Coulomb'), ...
    fullfile(worker_functions_root, 'Core', 'Direct'), ...
    fullfile(worker_functions_root, 'Core', 'Exchange')};
worker_path_command = cellfun(@(p) sprintf('addpath(''%s'', ''-begin'');', p), ...
    worker_paths, 'UniformOutput', false);
worker_path_command = [strjoin(worker_path_command, ' ') ...
    ' rehash; clear build_HBSE_flattened direct_coulomb_source_group;'];
pctRunOnAll(worker_path_command);
fprintf('[OPT  ] Parallel: attached HBSE source client=%s worker=%s\n', client_hbse_source, worker_hbse_source);
fprintf('[OPT  ] Parallel: attached direct source client=%s worker=%s\n', client_direct_source, worker_direct_source);
fprintf('[OPT  ] Parallel: worker function tree=%s\n', worker_functions_root);
end
