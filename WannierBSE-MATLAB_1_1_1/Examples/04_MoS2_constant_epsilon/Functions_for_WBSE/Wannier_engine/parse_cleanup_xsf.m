function cleanup = parse_cleanup_xsf(value)
%==========================================================================
% PARSE_CLEANUP_XSF
%
% DESCRIPTION:
%   Strictly parses the cleanup_xsf user option. Only the case-insensitive,
%   trimmed text value "true" enables cleanup; all other values disable it.
%
% INPUT ARGUMENTS:
%   value : raw cleanup_xsf parameter value.
%
% OUTPUT:
%   cleanup : logical scalar indicating whether cleanup is enabled.
%==========================================================================

cleanup = false;
if nargin < 1
    return;
end
if ischar(value) || isstring(value)
    cleanup = strcmpi(strtrim(char(value)), 'true');
end
end
