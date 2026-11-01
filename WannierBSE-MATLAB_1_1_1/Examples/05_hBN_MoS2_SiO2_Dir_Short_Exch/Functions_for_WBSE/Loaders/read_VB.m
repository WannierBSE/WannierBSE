function [Ev, Cv] = read_VB(bandSource, FunctionPath, Nv, N_kpt, e, NB)
%==========================================================================
% READ_VB
%
% DESCRIPTION:
%   Loads selected valence-band energies and TB eigenvectors from WBSE-compatible band files.
%
% INPUT ARGUMENTS:
%   bandSource : directory containing valence-band files.
%   FunctionPath : retained legacy argument; not used by the current loader.
%   Nv : number of valence bands.
%   N_kpt : number of k-points.
%   e : energy conversion factor.
%   NB : number of Wannier basis coefficients.
%
% OUTPUT:
%   Ev : N_kpt-by-Nv valence energies.
%   Cv : N_kpt-by-NB-by-Nv complex TB coefficients.
%==========================================================================

    Ev = zeros(N_kpt, Nv);
    Cv = zeros(N_kpt, NB, Nv);
    
    for i = 1:Nv
        % Priority 1: Modern Naming (v1_TB.txt)
        VB_NAME = fullfile(bandSource, sprintf('v%d_TB.txt', i));
        
        % Priority 2: Cache Naming (v1_TB_WBSE.txt)
        if ~isfile(VB_NAME)
            VB_NAME = fullfile(bandSource, sprintf('v%d_TB_WBSE.txt', i));
        end
        
        % Final Validation before loading
        if ~isfile(VB_NAME)
            abs_path = char(java.io.File(VB_NAME).getCanonicalPath());
            error('read_VB:FileNotFound', 'Error: Band data file not found for valence band %d.\nExpected path: %s', i, abs_path);
        end
        
        % Load the raw numeric data
        VB_data_raw = importdata(VB_NAME);
        if isstruct(VB_data_raw), VB_data_raw = VB_data_raw.data; end
        
        % USE YOUR ORIGINAL HELPER FUNCTION
        [Ev(:,i), Cv(:,:,i)] = read_TB_data(N_kpt, VB_data_raw, e, NB);
    end
end
