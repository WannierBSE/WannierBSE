function configure_parallel_runtime()
%==========================================================================
% CONFIGURE_PARALLEL_RUNTIME
%
% DESCRIPTION:
%   Sets client-side native-library thread limits before creation of a
%   MATLAB process pool. Worker-local configuration is performed separately
%   after pool creation by ensure_parallel_pool.
%
% INPUT ARGUMENTS:
%   None.
%
% OUTPUT:
%   None. Thread-limit environment variables are configured in the client.
%==========================================================================

setenv('OMP_NUM_THREADS', '1');
setenv('MKL_NUM_THREADS', '1');
setenv('OPENBLAS_NUM_THREADS', '1');
setenv('MKL_DYNAMIC', 'FALSE');

fprintf('[OPT  ] Parallel: OMP_NUM_THREADS=%s\n', getenv('OMP_NUM_THREADS'));
fprintf('[OPT  ] Parallel: MKL_NUM_THREADS=%s\n', getenv('MKL_NUM_THREADS'));
fprintf('[OPT  ] Parallel: OPENBLAS_NUM_THREADS=%s\n', getenv('OPENBLAS_NUM_THREADS'));
fprintf('[OPT  ] Parallel: MKL_DYNAMIC=%s\n', getenv('MKL_DYNAMIC'));
fprintf('[OPT  ] Parallel: internal worker-thread policy will be verified after pool creation.\n');
end
