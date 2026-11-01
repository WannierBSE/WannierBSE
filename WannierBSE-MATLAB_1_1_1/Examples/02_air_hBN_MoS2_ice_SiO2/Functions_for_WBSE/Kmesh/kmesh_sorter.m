function [k_sorted, index_map, header_out] = kmesh_sorter(input_data, N_de_in, N_co_in)
%==========================================================================
% KMESH_SORTER
%
% DESCRIPTION:
%   Sorts a mixed dense/coarse k-mesh into a deterministic WBSE order while preserving the dense and
%   coarse partitions.
%
% INPUT ARGUMENTS:
%   input_data : unsorted k-point table.
%   N_de_in : number of dense k-points.
%   N_co_in : number of coarse k-points.
%
% OUTPUT:
%   k_sorted : sorted k-point table.
%   index_map : sorted-to-original row map.
%   header_out : point-count metadata for downstream file writers.
%==========================================================================

    if ischar(input_data) || isstring(input_data)
        % --- FILE MODE ---
        if ~isfile(input_data), error('WannierBSE:Sorter:FileNotFound', 'File not found: %s', input_data); end
        
        fid = fopen(input_data, 'r');
        header_out = fscanf(fid, '%f', 5);
        % Header extraction: [1:Dense_range, 2:Nk_coarse, 3:Nk_dense, 4:N_co, 5:N_de]
        N_co = header_out(4);
        N_de = header_out(5);
        
        k_raw_full = fscanf(fid, '%f', [3, inf]);
        k_raw = k_raw_full(1:2, :).'; % Extract kx, ky
        fclose(fid);
        
        % CRITICAL VALIDATION: Check header vs. physical rows
        ActualRows = size(k_raw, 1);
        if (N_de + N_co) ~= ActualRows
            error('WannierBSE:ValidationFailed', ...
                ['[ERROR] K-mesh Validation Failed: Header defines N_de+N_co = %d, ' ...
                 'but the file contains %d points. Please check %s.'], ...
                (N_de + N_co), ActualRows, input_data);
        end
    else
        % --- DIRECT MODE ---
        k_raw = input_data(:, 1:2);
        N_de = N_de_in;
        N_co = N_co_in;
        header_out = []; % Not used in direct mode
    end

    % =========================================================================
    % 2. DENSITY-BASED PARTITIONING
    % =========================================================================
    dist_matrix = pdist2(k_raw, k_raw);
    dist_matrix(dist_matrix == 0) = Inf;
    min_distances = min(dist_matrix).'; 

    [~, global_idx] = sort(min_distances, 'ascend');
    
    idx_dense_pool  = global_idx(1:N_de);
    idx_coarse_pool = global_idx(N_de+1:end);
    
    % =========================================================================
    % 3. INTERNAL COORDINATE SORTING (Kx -> Ky)
    % =========================================================================
    % Dense Block
    k_dense_pool = k_raw(idx_dense_pool, :);
    [~, local_idx_dense] = sortrows(k_dense_pool, [1 2]);
    
    % Coarse Block
    k_coarse_pool = k_raw(idx_coarse_pool, :);
    [~, local_idx_coarse] = sortrows(k_coarse_pool, [1 2]);

    % Final Mapping
    final_idx_dense  = idx_dense_pool(local_idx_dense);
    final_idx_coarse = idx_coarse_pool(local_idx_coarse);
    
    index_map = [final_idx_dense; final_idx_coarse];
    k_sorted  = k_raw(index_map, :);

    % =========================================================================
    % 4. DIAGNOSTICS
    % =========================================================================
    dist_final = pdist2(k_sorted, k_sorted);
    dist_final(dist_final == 0) = Inf;
    final_spacing = min(dist_final).';
    contrast_ratio = final_spacing(N_de + 1) / final_spacing(N_de);
    
end
