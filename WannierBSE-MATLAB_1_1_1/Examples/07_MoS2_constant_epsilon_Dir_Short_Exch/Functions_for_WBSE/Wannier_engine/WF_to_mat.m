%==========================================================================
% WF_TO_MAT
%
% DESCRIPTION:
%   Selects the production parallel Wannier preprocessing path or the
%   explicitly requested legacy diagnostic path.
%
% INPUT ARGUMENTS:
%   None. The optional base-workspace flag Use_Legacy_WF_Preprocessing
%   selects the legacy path.
%
% OUTPUT:
%   None. The selected preprocessing workflow writes WBSE Wannier caches.
%==========================================================================
use_legacy = false;
if evalin('base','exist(''Use_Legacy_WF_Preprocessing'',''var'')')
    use_legacy = evalin('base','Use_Legacy_WF_Preprocessing');
end
if use_legacy
    fprintf('[WARN ] Wannier: using retained legacy preprocessing path.\n');
    WF_to_mat_legacy;
else
    WF_to_mat_parallel;
end
