function statuses = verify_parallel_runtime(pool, requestedWorkers, stageName)
%==========================================================================
% VERIFY_PARALLEL_RUNTIME
%
% DESCRIPTION:
%   Configures and validates every worker in an existing MATLAB process
%   pool.
%
% INPUT ARGUMENTS:
%   pool : valid local process-pool object.
%   requestedWorkers : requested worker count used for diagnostics.
%   stageName : optional stage label used in diagnostics.
%
% OUTPUT:
%   statuses : cell array of worker configuration status structures.
%==========================================================================

if nargin < 3 || isempty(stageName)
    stageName = 'parallel';
end
if isempty(pool) || ~isvalid(pool)
    error('WBSE:ParallelPoolUnavailable', 'No valid process pool is available for stage %s.', stageName);
end

statuses = cell(1, pool.NumWorkers);
spmd
    localStatus = configure_parallel_worker();
end
for lab = 1:pool.NumWorkers
    statuses{lab} = localStatus{lab};
end

ok = true;
for lab = 1:numel(statuses)
    s = statuses{lab};
    ok = ok && isstruct(s) && isfield(s, 'success') && s.success;
end
if ~ok
    for lab = 1:numel(statuses)
        s = statuses{lab};
        if isstruct(s) && isfield(s, 'error_message') && ~isempty(s.error_message)
            fprintf(2, '[ERROR] Parallel worker %d: %s\n', lab, s.error_message);
        end
    end
    error('WBSE:ParallelRuntimeUnverified', ...
          'Worker runtime policy could not be verified for stage %s.', stageName);
end

threads = cellfun(@(s) s.maxNumCompThreads, statuses);
fprintf('[OPT  ] Parallel: stage=%s requested_workers=%d\n', stageName, requestedWorkers);
fprintf('[OPT  ] Parallel: stage=%s actual_workers=%d\n', stageName, pool.NumWorkers);
fprintf('[OPT  ] Parallel: stage=%s computational_threads_per_worker=%d\n', stageName, threads(1));
fprintf('[OPT  ] Parallel: stage=%s runtime_policy=verified\n', stageName);
end
