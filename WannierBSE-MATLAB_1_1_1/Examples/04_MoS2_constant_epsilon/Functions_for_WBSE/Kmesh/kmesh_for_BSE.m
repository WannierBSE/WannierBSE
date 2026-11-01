function kmesh_for_BSE()
%==========================================================================
% KMESH_FOR_BSE
%
% DESCRIPTION:
%   Orchestrates generation, deterministic sorting, and export of the non-uniform D3h-compatible k-mesh
%   used by the BSE calculation.
%
% INPUT ARGUMENTS:
%   None. Configuration is loaded from ../Parameters/kmesh_control.txt.
%
% OUTPUT:
%   None. The generated mesh is written to ../Precomputed_data/kmesh_WBSE.txt.
%==========================================================================

    output_path = '../Precomputed_data/kmesh_WBSE.txt';
    if isfile(output_path)
        disp(['[INFO] Existing k-mesh file detected: ' output_path]);
        return;
    end

    % 2. CONFIGURATION LOADING
    param_file = '../Parameters/kmesh_control.txt';
    if ~isfile(param_file)
        error('WannierBSE:FileNotFound', 'Configuration file not found: %s', param_file);
    end
    
    params = read_params(param_file);
    Nk_coarse   = params.Nk_coarse;
    Nk_dense    = params.Nk_dense;
    Dense_range = params.Dense_range;
    
    if Dense_range > Nk_coarse / 2
        error('WannierBSE:ValidationFailed', ...
              'Dense_range must be <= Nk_coarse/2 to prevent point duplication.');
    end

    % 3. LATTICE DATA ACQUISITION & ENGINE EXECUTION
    [a1, a2, ~, a_const] = read_structure();
    
    % Execute core geometric engine to obtain dimensionless coordinates
    [final_kmesh, total_coarse, total_dense, ~, raw_topology, raw_neighbor_pairs] = ...
        kmesh_hexagonal_engine(Nk_coarse, Nk_dense, Dense_range, a1, a2, a_const);

    % =========================================================================
    % NEW: INTEGRATION OF DETERMINISTIC SORTING
    % =========================================================================
    % Points are sorted before scaling to ensure dimensionless precision.
    [final_kmesh, index_map] = kmesh_sorter(final_kmesh, total_dense, total_coarse);
    [sorted_topology, sorted_neighbor_pairs] = sort_kmesh_topology(raw_topology, raw_neighbor_pairs, index_map);

    % 4. EXPORT TO KMESH.TXT (Physical Units Application)
    fid_out = fopen(output_path, 'w');
    if fid_out == -1
        error('WannierBSE:FileWriteError', 'Unable to open %s for writing.', output_path);
    end

    % Write Header Information
    fprintf(fid_out, '%d\n', Dense_range);
    fprintf(fid_out, '%d\n', Nk_coarse);
    fprintf(fid_out, '%d\n', Nk_dense);
    fprintf(fid_out, '%d\n', total_coarse); 
    fprintf(fid_out, '%d\n', total_dense);

    % Scale factor for physical K-space conversion (1/Angstrom)
    scale_factor = pi / a_const;

    % Export 3D k-points with high-fidelity precision
    for i = 1:size(final_kmesh, 1)
        kx_scaled = final_kmesh(i,1) * scale_factor;
        ky_scaled = final_kmesh(i,2) * scale_factor;
        fprintf(fid_out, '%20.16f    %20.16f     %20.16f\n', kx_scaled, ky_scaled, 0.0);
    end
    fclose(fid_out);

    write_kmesh_topology_sidecars(sorted_topology, sorted_neighbor_pairs);
end

function [sorted_topology, sorted_neighbor_pairs] = sort_kmesh_topology(raw_topology, raw_neighbor_pairs, index_map)
%==========================================================================
% SORT_KMESH_TOPOLOGY
%
% DESCRIPTION:
%   Applies the k-mesh sort permutation to generator topology metadata and neighbor-pair indices.
%
% OUTPUT:
%   sorted_topology : topology records aligned with final kmesh_WBSE rows.
%   sorted_neighbor_pairs : unique sorted neighbor pairs in final row indexing.
%==========================================================================

    sorted_topology = raw_topology(index_map);
    inverse_index_map = zeros(numel(index_map), 1);
    inverse_index_map(index_map) = (1:numel(index_map)).';

    for k_index = 1:numel(sorted_topology)
        sorted_topology(k_index).final_k_index = k_index;
        sorted_topology(k_index).sorted_output_index = k_index;
    end

    if isempty(raw_neighbor_pairs)
        sorted_neighbor_pairs = zeros(0, 2);
    else
        sorted_neighbor_pairs = inverse_index_map(raw_neighbor_pairs);
        sorted_neighbor_pairs = sort(sorted_neighbor_pairs, 2);
        sorted_neighbor_pairs = unique(sorted_neighbor_pairs, 'rows');
    end
end

function write_kmesh_topology_sidecars(sorted_topology, sorted_neighbor_pairs)
%==========================================================================
% WRITE_KMESH_TOPOLOGY_SIDECARS
%
% DESCRIPTION:
%   Writes temporary topology and adjacency sidecar files for TB band tracking on the generated k-mesh.
%==========================================================================

    temp_path = '../Precomputed_data/TB_data/temp';
    if ~exist(temp_path, 'dir')
        mkdir(temp_path);
    end

    topology_path = fullfile(temp_path, 'kmesh_WBSE_topology.txt');
    topology_tmp = [topology_path, '.tmp'];
    fid = fopen(topology_tmp, 'w');
    if fid == -1
        error('WannierBSE:FileWriteError', 'Unable to open %s for writing.', topology_tmp);
    end

    try
        fprintf(fid, '# k_index region_type region_id sector_id template_id patch_id triangle_id local_i local_j triangle_orientation sorted_output_index\n');
        for k_index = 1:numel(sorted_topology)
            t = sorted_topology(k_index);
            fprintf(fid, '%8d %-6s %-18s %3d %-2s %-13s %6d %6d %6d %-4s %8d\n', ...
                    t.final_k_index, t.region_type, t.region_id, t.sector_id, ...
                    t.template_id, t.patch_id, t.triangle_id, t.local_i, t.local_j, ...
                    t.triangle_orientation, t.sorted_output_index);
        end
        fclose(fid);
        movefile(topology_tmp, topology_path, 'f');
    catch ME
        fclose(fid);
        if isfile(topology_tmp)
            delete(topology_tmp);
        end
        rethrow(ME);
    end

    neighbors_path = fullfile(temp_path, 'kmesh_WBSE_neighbors.txt');
    neighbors_tmp = [neighbors_path, '.tmp'];
    fid = fopen(neighbors_tmp, 'w');
    if fid == -1
        error('WannierBSE:FileWriteError', 'Unable to open %s for writing.', neighbors_tmp);
    end

    try
        fprintf(fid, '# k_index neighbor_1 neighbor_2 neighbor_3 neighbor_4 neighbor_5 neighbor_6\n');
        adjacency = build_neighbor_columns(numel(sorted_topology), sorted_neighbor_pairs, 6);
        for k_index = 1:size(adjacency, 1)
            fprintf(fid, '%8d %8d %8d %8d %8d %8d %8d\n', k_index, adjacency(k_index, :));
        end
        fclose(fid);
        movefile(neighbors_tmp, neighbors_path, 'f');
    catch ME
        fclose(fid);
        if isfile(neighbors_tmp)
            delete(neighbors_tmp);
        end
        rethrow(ME);
    end
end

function adjacency = build_neighbor_columns(n_points, neighbor_pairs, max_neighbors)
%==========================================================================
% BUILD_NEIGHBOR_COLUMNS
%
% DESCRIPTION:
%   Converts an unordered neighbor-pair list into fixed-width adjacency columns.
%
% OUTPUT:
%   adjacency : integer neighbor table padded with zeros.
%==========================================================================

    adjacency = zeros(n_points, max_neighbors);
    counts = zeros(n_points, 1);
    for e = 1:size(neighbor_pairs, 1)
        i = neighbor_pairs(e, 1);
        j = neighbor_pairs(e, 2);
        counts(i) = counts(i) + 1;
        counts(j) = counts(j) + 1;
        if counts(i) > max_neighbors || counts(j) > max_neighbors
            error('WannierBSE:KmeshTopology', 'More than %d neighbors for a k-point.', max_neighbors);
        end
        adjacency(i, counts(i)) = j;
        adjacency(j, counts(j)) = i;
    end
end
