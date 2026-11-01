function [ok, reason] = validate_wf_cache(cache_path)
%==========================================================================
% VALIDATE_WF_CACHE
%
% DESCRIPTION:
%   Validates the required files and metadata of a production Wannier cache
%   without loading the large wavefunction arrays.
%
% INPUT ARGUMENTS:
%   cache_path : path to the Wannier cache directory.
%
% OUTPUT:
%   ok : logical scalar indicating whether the cache is valid.
%   reason : diagnostic text describing a validation failure.
%==========================================================================
required = {'WF_centers_WBSE.txt','WF_up.mat','WF_down.mat','r.mat','GridInfo.txt','WF_AtomPosition.txt'};
ok = isfolder(cache_path); reason = '';
if ~ok, reason = 'cache directory missing'; return; end
for k=1:numel(required)
    if ~isfile(fullfile(cache_path,required{k})), ok=false; reason=['missing ' required{k}]; return; end
end
try
    wu=whos('-file',fullfile(cache_path,'WF_up.mat'),'W_up');
    wd=whos('-file',fullfile(cache_path,'WF_down.mat'),'W_down');
    if isempty(wu)||isempty(wd)||~strcmp(wu.class,'double')||~strcmp(wd.class,'double')||~wu.complex||~wd.complex||~isequal(wu.size,[4199040 22])||~isequal(wd.size,[4199040 22])
        ok=false; reason='WF matrix shape or variable mismatch'; return;
    end
catch err
    ok=false; reason=['MAT validation failed: ' err.message]; return;
end
end

