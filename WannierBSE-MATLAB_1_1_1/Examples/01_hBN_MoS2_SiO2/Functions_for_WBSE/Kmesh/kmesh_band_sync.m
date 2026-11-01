function kmesh_band_sync(SourcePath, DestPath, index_map, Nv, Nc, NB, k_fixed)
%==========================================================================
% KMESH_BAND_SYNC
%
% DESCRIPTION:
%   Permutes external TB band files into the sorted WBSE k-mesh order and injects the final physical k
%   coordinates.
%
% INPUT ARGUMENTS:
%   SourcePath : directory containing v*_TB.txt and c*_TB.txt files.
%   DestPath : directory receiving v*_TB_WBSE.txt and c*_TB_WBSE.txt files.
%   index_map : sorted-to-original k-point permutation.
%   Nv, Nc : number of selected valence and conduction bands.
%   NB : number of Wannier basis functions per band block.
%   k_fixed : final sorted k-point coordinates.
%
% OUTPUT:
%   None. The synchronized band files are written to DestPath.
%==========================================================================

    v_files = arrayfun(@(n) sprintf('v%d_TB.txt', n), 1:Nv, 'UniformOutput', false);
    c_files = arrayfun(@(n) sprintf('c%d_TB.txt', n), 1:Nc, 'UniformOutput', false);
    all_files = [v_files, c_files];
    
    N_kpt = length(index_map);
    lines_per_block = 2 + NB;

    for f = 1:length(all_files)
        fname = all_files{f};
        input_path  = fullfile(SourcePath, fname);
        [~, name_part, ext_part] = fileparts(fname);
        output_fname = [name_part, '_WBSE', ext_part]; % Result: v1_TB_WBSE.txt
        output_path = fullfile(DestPath, output_fname);
                
        if ~isfile(input_path)
            warning('WannierBSE:Sync:MissingFile', 'Band file not found: %s. Skipping.', fname);
            continue;
        end
        
              
        % --- Step 1: Ingest Data Blocks ---
        fid_in = fopen(input_path, 'r');
        % Read all lines as a cell array for robust block handling
        raw_data = textscan(fid_in, '%s', 'Delimiter', '\n');
        fclose(fid_in);
        lines = raw_data{1};
        
        % Validate file length
        if length(lines) < N_kpt * lines_per_block
             error('WannierBSE:Sync:SizeMismatch', ...
                 'File %s size does not match N_kpt * (2+NB).', fname);
        end
        
        % --- Step 2: Permute and Inject K-coordinates ---
        % Pre-allocate cell array for sorted blocks
        sorted_lines = cell(N_kpt * lines_per_block, 1);
        
        for i = 1:N_kpt
            % Find original position from the map
            original_idx = index_map(i);
            start_row = (original_idx - 1) * lines_per_block + 1;
            
            % Target positions in the new file
            target_start = (i - 1) * lines_per_block + 1;
            
            % Inject Option B: Use sorted physical K-coordinates
            % Column 1: K-vector Line (Updated)
            sorted_lines{target_start} = sprintf('%20.16f    %20.16f     %20.16f', ...
                k_fixed(i,1), k_fixed(i,2), 0.0);
            
            % Columns 2 to End: Copy Energy and Eigenvectors exactly
            sorted_lines(target_start+1 : target_start + lines_per_block - 1) = ...
                lines(start_row+1 : start_row + lines_per_block - 1);
        end
        
        % --- Step 3: High-Fidelity Export ---
        fid_out = fopen(output_path, 'w');
        fprintf(fid_out, '%s\n', sorted_lines{:});
        fclose(fid_out);
    end
end
