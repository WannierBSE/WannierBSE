function token = opt_wf_timing(action, token, phase, scope, items, input_shape)
%==========================================================================
% OPT_WF_TIMING
%
% DESCRIPTION:
%   Records optional logging-only timing information for Wannier
%   preprocessing. It does not alter numerical values, ordering, or files.
%
% INPUT ARGUMENTS:
%   action : 'start' or 'stop'.
%   token : timer token returned by the start action.
%   phase, scope, items, input_shape : optional diagnostic labels.
%
% OUTPUT:
%   token : timer token for start, or empty after stop.
%==========================================================================
    if nargin < 2, token = []; end
    if nargin < 3, phase = 'unclassified_overhead'; end
    if nargin < 4 || isempty(scope), scope = 'exclusive'; end
    if nargin < 5 || isempty(items), items = '1'; end
    if nargin < 6 || isempty(input_shape), input_shape = 'NA'; end
    switch lower(action)
        case 'start'
            token = tic;
        case 'stop'
            elapsed = toc(token);
            fprintf('[OPT ] WF implementation=MATLAB phase=%s timing_scope=%s time_s=%.6f items=%s input_shape=%s\n', ...
                phase, scope, elapsed, items, input_shape);
            token = [];
        otherwise
            error('opt_wf_timing:InvalidAction', 'Unknown timing action: %s', action);
    end
end

