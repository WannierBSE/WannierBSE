function [Ec, Cc] = read_CB(bandSource, FunctionPath, Nc, N_kpt, e, NB)
%==========================================================================
% READ_CB
%
% DESCRIPTION:
%   Loads selected conduction-band energies and TB eigenvectors from WBSE-compatible band files.
%
% INPUT ARGUMENTS:
%   bandSource : directory containing conduction-band files.
%   FunctionPath : retained legacy argument; not used by the current loader.
%   Nc : number of conduction bands.
%   N_kpt : number of k-points.
%   e : energy conversion factor.
%   NB : number of Wannier basis coefficients.
%
% OUTPUT:
%   Ec : N_kpt-by-Nc conduction energies.
%   Cc : N_kpt-by-NB-by-Nc complex TB coefficients.
%==========================================================================

    Ec = zeros(N_kpt, Nc);
    Cc = zeros(N_kpt, NB, Nc);
    
    for i = 1:Nc
        % Priority 1: Modern Naming (c1_TB.txt)
        CB_NAME = fullfile(bandSource, sprintf('c%d_TB.txt', i));
        
        % Priority 2: Cache Naming (c1_TB_WBSE.txt)
        if ~isfile(CB_NAME)
            CB_NAME = fullfile(bandSource, sprintf('c%d_TB_WBSE.txt', i));
        end
        
        % Final Validation before loading
        if ~isfile(CB_NAME)
            abs_path = char(java.io.File(CB_NAME).getCanonicalPath());
            error('read_CB:FileNotFound', 'Error: Band data file not found for conduction band %d.\nExpected path: %s', i, abs_path);
        end
        
        CB_data_raw = importdata(CB_NAME);
        if isstruct(CB_data_raw), CB_data_raw = CB_data_raw.data; end
        
        [Ec(:,i), Cc(:,:,i)] = read_TB_data(N_kpt, CB_data_raw, e, NB);
    end
end
