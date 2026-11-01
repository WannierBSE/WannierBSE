function status = configure_parallel_worker()
%==========================================================================
% CONFIGURE_PARALLEL_WORKER
%
% DESCRIPTION:
%   Configures one MATLAB process worker and reports its thread settings and
%   process metadata.
%
% INPUT ARGUMENTS:
%   None. Worker identity and environment are obtained from MATLAB.
%
% OUTPUT:
%   status : structure containing worker configuration and validation fields.
%==========================================================================

status = struct('success', false, 'worker_index', [], 'pid', [], ...
                'maxNumCompThreads', NaN, 'OMP_NUM_THREADS', '', ...
                'MKL_NUM_THREADS', '', 'OPENBLAS_NUM_THREADS', '', ...
                'MKL_DYNAMIC', '', 'error_message', '');
try
    setenv('OMP_NUM_THREADS', '1');
    setenv('MKL_NUM_THREADS', '1');
    setenv('OPENBLAS_NUM_THREADS', '1');
    setenv('MKL_DYNAMIC', 'FALSE');

    maxNumCompThreads(1);
    status.maxNumCompThreads = maxNumCompThreads;
    status.worker_index = labindex;
    try
        status.pid = feature('getpid');
    catch
        status.pid = [];
    end
    status.OMP_NUM_THREADS = getenv('OMP_NUM_THREADS');
    status.MKL_NUM_THREADS = getenv('MKL_NUM_THREADS');
    status.OPENBLAS_NUM_THREADS = getenv('OPENBLAS_NUM_THREADS');
    status.MKL_DYNAMIC = getenv('MKL_DYNAMIC');
    status.success = status.maxNumCompThreads == 1 && ...
        strcmp(status.OMP_NUM_THREADS, '1') && ...
        strcmp(status.MKL_NUM_THREADS, '1') && ...
        strcmp(status.OPENBLAS_NUM_THREADS, '1') && ...
        strcmpi(status.MKL_DYNAMIC, 'FALSE');
    if ~status.success
        status.error_message = 'Worker settings did not match the requested policy.';
    end
catch ME
    status.error_message = ME.message;
end
end
