function [fP, aT] = resolve_data_path(fN)
%==========================================================================
% RESOLVE_DATA_PATH
%
% DESCRIPTION:
%   Resolves a WBSE data file name against standard user-input and precomputed-data locations.
%
% INPUT ARGUMENTS:
%   fN : file name or path requested by the caller.
%
% OUTPUT:
%   fP : resolved file path.
%   aT : true when the resolved file is available.
%==========================================================================

    uF = fullfile('User_input', fN);
    if exist(uF, 'file')
        fP = uF;
        aT = 'LOAD ';
        return;
    end

    [~, n, e] = fileparts(fN);
    cF = fullfile('Precomputed_data', [n, '_WBSE', e]);
    if exist(cF, 'file')
        fP = cF;
        aT = 'CACHE';
        return;
    end

    fP = '';
    aT = 'GEN  ';
end
