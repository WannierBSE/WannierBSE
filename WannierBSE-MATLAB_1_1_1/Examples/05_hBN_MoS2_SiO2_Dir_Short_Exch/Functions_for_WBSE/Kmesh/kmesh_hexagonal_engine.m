function [final_kmesh, total_coarse, total_dense, orientation, topology, neighbor_pairs] = kmesh_hexagonal_engine( ...
    Nk_coarse, Nk_dense, Dense_range, a1, a2, a_const)
%==========================================================================
% KMESH_HEXAGONAL_ENGINE
%
% DESCRIPTION:
%   Generates the dimensionless dual-resolution hexagonal k-mesh by assembling triangular sectors with D3h-compatible symmetry.
%
% INPUT ARGUMENTS:
%   Nk_coarse : coarse mesh resolution per generator direction.
%   Nk_dense : dense valley mesh resolution.
%   Dense_range : coarse-grid range replaced by dense patches near valleys.
%   a1, a2 : real-space lattice vectors used for orientation checks.
%   a_const : lattice constant retained for caller-side scaling.
%
% OUTPUT:
%   final_kmesh : dimensionless k-point coordinates before physical scaling.
%   total_coarse, total_dense : point counts by mesh region.
%   orientation : detected hexagonal lattice orientation.
%   topology : generator metadata for each raw k-point.
%   neighbor_pairs : generator-local adjacent-triangle pairs.
%==========================================================================

    orientation = get_hex_orientation(a1, a2);

    % Define Basis Vectors U, V, and W (Dimensionless/Normalized)
    if strcmp(orientation, 'Horizontal')
        U = [2/sqrt(3), 2/3];
        V = [2/sqrt(3), -2/3];
        W = [0, -4/3];
    elseif strcmp(orientation, 'Vertical')
        U = [2/3, 2/sqrt(3)];
        V = [-2/3, 2/sqrt(3)];
        W = [-4/3, 0];
    else
        error('WannierBSE:LatticeError', 'Unsupported orientation: %s.', orientation);
    end

    % Geometric Increments (Coarse)
    u = U / Nk_coarse;
    v = V / Nk_coarse;
    w = W / Nk_coarse;

    % Geometric Increments (Dense)
    u_d = (U * (Dense_range / Nk_coarse)) / Nk_dense;
    v_d = (V * (Dense_range / Nk_coarse)) / Nk_dense;
    w_d = (W * (Dense_range / Nk_coarse)) / Nk_dense;

    % Local Origin Shifts for Dense Clusters
    Shift_U = U * (1 - Dense_range / Nk_coarse);
    Shift_V = V * (1 - Dense_range / Nk_coarse);
    Shift_W = W * (1 - Dense_range / Nk_coarse);

    limit_lower = Nk_coarse - Dense_range;

    % --- Slice 1: Segment Gamma-U-V ---
    [k_low1, m_low1] = triangular_kmesh(0, Nk_coarse-1, u, v, [0,0], limit_lower, 'lower', 'coarse', 'UV', 'coarse_lower', 'coarse');
    [k_up1,  m_up1]  = triangular_kmesh(0, Nk_coarse-1, u, v, [0,0], limit_lower, 'upper', 'coarse', 'UV', 'coarse_upper', 'coarse');
    [k_dU1,  m_dU1]  = triangular_kmesh(0, Nk_dense-1, u_d, v_d, Shift_U, [], 'all', 'dense', 'UV', 'dense_U', 'dense_U');
    [k_dV1,  m_dV1]  = triangular_kmesh(0, Nk_dense-1, u_d, v_d, Shift_V, [], 'all', 'dense', 'UV', 'dense_V', 'dense_V');
    master_UV = [k_low1; k_up1; k_dU1; k_dV1];
    meta_UV = [m_low1; m_up1; m_dU1; m_dV1];

    % --- Slice 2: Segment Gamma-V-W ---
    [k_low2, m_low2] = triangular_kmesh(0, Nk_coarse-1, v, w, [0,0], limit_lower, 'lower', 'coarse', 'VW', 'coarse_lower', 'coarse');
    [k_up2,  m_up2]  = triangular_kmesh(0, Nk_coarse-1, v, w, [0,0], limit_lower, 'upper', 'coarse', 'VW', 'coarse_upper', 'coarse');
    [k_dV2,  m_dV2]  = triangular_kmesh(0, Nk_dense-1, v_d, w_d, Shift_V, [], 'all', 'dense', 'VW', 'dense_V', 'dense_V');
    [k_dW2,  m_dW2]  = triangular_kmesh(0, Nk_dense-1, v_d, w_d, Shift_W, [], 'all', 'dense', 'VW', 'dense_W', 'dense_W');
    master_VW = [k_low2; k_up2; k_dV2; k_dW2];
    meta_VW = [m_low2; m_up2; m_dV2; m_dW2];

    % Symmetry Operators (Reflection Sign Flips)
    if strcmp(orientation, 'Vertical')
        M = { [1, 1], [1, 1], [1, -1], [1, -1], [-1, -1], [-1, 1] };
    else
        M = { [1, 1], [1, 1], [-1, 1], [-1, 1], [-1, -1], [1, -1] };
    end

    % Hexagonal Assembly (6-Slice Reconstruction)
    S = { master_UV, master_VW, master_VW, master_UV, master_VW, master_VW };
    SM = { meta_UV, meta_VW, meta_VW, meta_UV, meta_VW, meta_VW };
    final_kmesh = [];
    topology = empty_topology();
    raw_index = 0;
    for sector = 1:6
        sector_k = S{sector} .* M{sector};
        sector_meta = SM{sector};
        final_kmesh = [final_kmesh; sector_k]; %#ok<AGROW>
        for n = 1:numel(sector_meta)
            raw_index = raw_index + 1;
            sector_meta(n).raw_index = raw_index;
            sector_meta(n).sector_id = sector;
            sector_meta(n).region_id = sprintf('S%d_%s_%s', sector, sector_meta(n).template_id, sector_meta(n).patch_id);
            topology = [topology; sector_meta(n)]; %#ok<AGROW>
        end
    end

    % Statistical Point Counts
    num_coarse_UV = size([k_low1; k_up1], 1);
    num_dense_UV  = size([k_dU1; k_dV1], 1);
    num_coarse_VW = size([k_low2; k_up2], 1);
    num_dense_VW  = size([k_dV2; k_dW2], 1);

    total_coarse = (num_coarse_UV * 2) + (num_coarse_VW * 4);
    total_dense  = (num_dense_UV * 2) + (num_dense_VW * 4);

    neighbor_pairs = build_generator_neighbors(topology);
end

function orientation = get_hex_orientation(a1, a2)
%==========================================================================
% GET_HEX_ORIENTATION
%
% DESCRIPTION:
%   Classifies the hexagonal lattice orientation from the real-space primitive vectors.
%
% OUTPUT:
%   orientation : Horizontal, Vertical, or a validation status string.
%==========================================================================

    a = norm(a1); tol = 1e-2;
    if abs(a - norm(a2)) > tol || ~any(abs(rad2deg(acos(dot(a1,a2)/(a*a))) - [60, 120]) < tol)
        orientation = 'Not Hexagonal'; return;
    end
    test_list = {a1, a2, a1+a2, a1-a2};
    for i = 1:length(test_list)
        v = test_list{i};
        if abs(v(1)) < tol && abs(abs(v(2)) - a) < tol
            orientation = 'Horizontal'; return;
        elseif abs(v(2)) < tol && abs(abs(v(1)) - a) < tol
            orientation = 'Vertical'; return;
        end
    end
    orientation = 'Neither';
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: TRIANGULAR_KMESH
%
% PURPOSE:
%   Internal helper used by kmesh_hexagonal_engine and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function [k_list, mesh_lines] = triangular_kmesh( ...
    N_start, N_end, u, v, shift, limit, mode, region_type, template_id, patch_id, mesh_group)
%==========================================================================
% TRIANGULAR_KMESH
%
% DESCRIPTION:
%   Generates centroid-sampled triangular sub-mesh points and topology records for one mesh patch.
%
% OUTPUT:
%   k_list : generated two-dimensional k coordinates.
%   mesh_lines : topology records for generated triangles.
%==========================================================================

    k_list = [];
    mesh_lines = empty_topology();
    triangle_id = 0;
    for i = N_start : N_end
        for j = 0:i
            % Downward Triangle Centroid Calculation
            P1 = (i-j)*u + j*v + shift;
            P2 = (i-j+1)*u + j*v + shift;
            P3 = (i-j)*u + (j+1)*v + shift;
            c_up = (P1 + P2 + P3) / 3.0;
            vertices_up = [i-j, j; i-j+1, j; i-j, j+1];

            keep_up = false;
            if strcmp(mode, 'all'), keep_up = true;
            elseif strcmp(mode, 'lower'), if i < limit, keep_up = true; end
            elseif strcmp(mode, 'upper'), if i >= limit && ~(j > limit - 1 || (i-j) > limit - 1), keep_up = true; end
            end
            if keep_up
                triangle_id = triangle_id + 1;
                k_list = [k_list; c_up]; %#ok<AGROW>
                mesh_lines = [mesh_lines; make_topology(region_type, template_id, patch_id, mesh_group, i, j, 'up', triangle_id, vertices_up)]; %#ok<AGROW>
            end

            % Upward Triangle Centroid Calculation
            if (i + 1 <= N_end)
                Pd1 = (i-j+1)*u + j*v + shift;
                Pd2 = (i-j)*u + (j+1)*v + shift;
                Pd3 = (i-j+1)*u + (j+1)*v + shift;
                c_down = (Pd1 + Pd2 + Pd3) / 3.0;
                vertices_down = [i-j+1, j; i-j, j+1; i-j+1, j+1];

                keep_down = false;
                if strcmp(mode, 'all'), keep_down = true;
                elseif strcmp(mode, 'lower'), if (i + 1 < limit), keep_down = true; end
                elseif strcmp(mode, 'upper'), if (i + 1 >= limit) && (j <= limit - 1) && ((i-j) <= limit - 1), keep_down = true; end
                end
                if keep_down
                    triangle_id = triangle_id + 1;
                    k_list = [k_list; c_down]; %#ok<AGROW>
                    mesh_lines = [mesh_lines; make_topology(region_type, template_id, patch_id, mesh_group, i, j, 'down', triangle_id, vertices_down)]; %#ok<AGROW>
                end
            end
        end
    end
end

function t = empty_topology()
%==========================================================================
% EMPTY_TOPOLOGY
%
% DESCRIPTION:
%   Returns an empty topology structure with the fields required by the k-mesh sidecar pipeline.
%
% OUTPUT:
%   t : empty topology structure array.
%==========================================================================

    t = struct('raw_index', {}, 'final_k_index', {}, 'sorted_output_index', {}, ...
               'region_type', {}, 'region_id', {}, 'sector_id', {}, ...
               'template_id', {}, 'patch_id', {}, 'mesh_group', {}, ...
               'triangle_id', {}, 'local_i', {}, 'local_j', {}, ...
               'triangle_orientation', {}, 'vertices', {});
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: MAKE_TOPOLOGY
%
% PURPOSE:
%   Internal helper used by kmesh_hexagonal_engine and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function t = make_topology( ...
    region_type, template_id, patch_id, mesh_group, local_i, local_j, orientation, triangle_id, vertices)
%==========================================================================
% MAKE_TOPOLOGY
%
% DESCRIPTION:
%   Constructs one topology record for a generated triangular k-mesh cell.
%
% OUTPUT:
%   t : populated topology structure.
%==========================================================================

    t = struct();
    t.raw_index = -1;
    t.final_k_index = -1;
    t.sorted_output_index = -1;
    t.region_type = region_type;
    t.region_id = '';
    t.sector_id = -1;
    t.template_id = template_id;
    t.patch_id = patch_id;
    t.mesh_group = mesh_group;
    t.triangle_id = triangle_id;
    t.local_i = local_i;
    t.local_j = local_j;
    t.triangle_orientation = orientation;
    t.vertices = vertices;
end

function neighbor_pairs = build_generator_neighbors(topology)
%==========================================================================
% BUILD_GENERATOR_NEIGHBORS
%
% DESCRIPTION:
%   Finds neighboring triangular cells by matching shared generator edges within each sector and mesh group.
%
% OUTPUT:
%   neighbor_pairs : two-column list of adjacent raw topology indices.
%==========================================================================

    edge_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
    pairs = zeros(0, 2);
    for idx = 1:numel(topology)
        vertices = topology(idx).vertices;
        edge_vertices = {vertices([1, 2], :), vertices([2, 3], :), vertices([1, 3], :)};
        for e = 1:3
            key = edge_key(topology(idx), edge_vertices{e});
            if isKey(edge_map, key)
                previous = edge_map(key);
                for p = previous(:).'
                    pairs = [pairs; p, idx]; %#ok<AGROW>
                end
                edge_map(key) = [previous(:); idx];
            else
                edge_map(key) = idx;
            end
        end
    end
    if isempty(pairs)
        neighbor_pairs = zeros(0, 2);
    else
        pairs = sort(pairs, 2);
        neighbor_pairs = unique(pairs, 'rows');
    end
end

function key = edge_key(meta, vertices)
%==========================================================================
% EDGE_KEY
%
% DESCRIPTION:
%   Builds a deterministic hash key for a topology edge.
%
% OUTPUT:
%   key : character key suitable for containers.Map lookup.
%==========================================================================

    vertices = sortrows(vertices, [1, 2]);
    key = sprintf('%d|%s|%s|%d,%d;%d,%d', meta.sector_id, meta.template_id, meta.mesh_group, ...
                  vertices(1,1), vertices(1,2), vertices(2,1), vertices(2,2));
end
