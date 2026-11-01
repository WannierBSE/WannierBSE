function parsed = read_xsf_wf_fast(file_path, method)
%==========================================================================
% READ_XSF_WF_FAST
%
% DESCRIPTION:
%   Candidate MATLAB-native XSF reader for bounded Wannier preprocessing
%   experiments. It reads one XSF file, returns production-compatible value
%   ordering, and writes no files.
%
% INPUT ARGUMENTS:
%   file_path : XSF file path.
%   method : optional reader mode: 'fscanf', 'textscan', or 'fread_sscanf'.
%
% OUTPUT:
%   parsed : structure containing XSF geometry, values, and timing metadata.
%
% NOTES:
%   This candidate is not called by the current production workflow.
%==========================================================================

    if nargin < 2 || isempty(method)
        method = 'fscanf';
    end
    file_path = char(file_path);
    method = lower(char(method));
    if ~ismember(method, {'fscanf', 'textscan', 'fread_sscanf'})
        error('WFOptimized:InvalidMethod', 'Unsupported candidate method: %s', method);
    end

    total_start = tic;
    fid = fopen(file_path, 'r');
    if fid < 0
        error('WFOptimized:OpenFailed', 'Cannot open XSF file: %s', file_path);
    end
    cleanup = onCleanup(@() wf_fast_close_file(fid));

    header_start = tic;
    [dims, origin, vectors, atoms, data_start_line] = wf_fast_read_header(fid, file_path);
    header_time = toc(header_start);
    expected_values = prod(dims);

    values_start = tic;
    switch method
        case 'fscanf'
            file_order_values = fscanf(fid, '%f', expected_values);
        case 'textscan'
            parsed_values = textscan(fid, '%f', expected_values, ...
                'Delimiter', ' ', 'MultipleDelimsAsOne', true, ...
                'ReturnOnError', false, 'EndOfLine', '\r\n');
            file_order_values = parsed_values{1};
        case 'fread_sscanf'
            numeric_text = fread(fid, Inf, '*char').';
            file_order_values = sscanf(numeric_text, '%f', expected_values);
    end
    values_time = toc(values_start);

    if numel(file_order_values) < expected_values
        error('WFOptimized:TooFewValues', ...
              '%s contains %d numeric grid values; expected %d.', ...
              file_path, numel(file_order_values), expected_values);
    elseif numel(file_order_values) > expected_values
        error('WFOptimized:TooManyValues', ...
              '%s contains %d numeric grid values; expected %d.', ...
              file_path, numel(file_order_values), expected_values);
    end

    postcheck_start = tic;
    extra_numeric_values = 0;
    found_end_marker = false;
    if strcmp(method, 'fread_sscanf')
        found_end_marker = contains(upper(numeric_text), 'END_DATAGRID_3D');
    else
        while true
            line = fgetl(fid);
            if ~ischar(line)
                break;
            end
            trimmed = strtrim(line);
            if contains(upper(trimmed), 'END_DATAGRID_3D')
                found_end_marker = true;
                break;
            end
            numeric_tail = sscanf(trimmed, '%f');
            extra_numeric_values = extra_numeric_values + numel(numeric_tail);
        end
    end
    postcheck_time = toc(postcheck_start);
    if extra_numeric_values > 0
        error('WFOptimized:TooManyValues', ...
              '%s contains at least %d extra numeric values before END_DATAGRID_3D.', ...
              file_path, extra_numeric_values);
    end
    if ~found_end_marker
        error('WFOptimized:MissingEndMarker', ...
              '%s has no END_DATAGRID_3D marker after the numeric block.', file_path);
    end

    reshape_start = tic;
    % Production test_* readers parse six values per text line, then apply
    % raw'(:). This reshape preserves that exact production ordering while
    % avoiding the production position1 array.
    if mod(expected_values, 6) ~= 0
        error('WFOptimized:UnexpectedRowWidth', ...
              '%s has %d values, not divisible by the production six-value row width.', ...
              file_path, expected_values);
    end
    values = reshape(file_order_values, 6, []);
    values = values(:);
    reshape_time = toc(reshape_start);

    if numel(values) ~= expected_values
        error('WFOptimized:InternalCountError', '%s candidate output count is invalid.', file_path);
    end

    parsed = struct();
    parsed.path = file_path;
    parsed.method = method;
    parsed.dims = dims;
    parsed.origin = origin;
    parsed.vectors = vectors;
    parsed.atoms = atoms;
    parsed.data_start_line = data_start_line;
    parsed.expected_values = expected_values;
    parsed.values = values;
    parsed.class = class(values);
    parsed.is_complex = ~isreal(values);
    parsed.finite = all(isfinite(values));
    parsed.min_value = min(values);
    parsed.max_value = max(values);
    parsed.norm = norm(values);
    parsed.timing = struct(...
        'header_s', header_time, ...
        'values_s', values_time, ...
        'postcheck_s', postcheck_time, ...
        'reshape_s', reshape_time, ...
        'total_s', toc(total_start));
end

function [dims, origin, vectors, atoms, data_start_line] = wf_fast_read_header(fid, file_path)
    dims = [];
    origin = [];
    vectors = zeros(3, 3);
    atoms = {};
    found_primcoord = false;
    found_grid = false;
    line_number = 0;

    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        line_number = line_number + 1;
        token = upper(strtrim(line));

        if strcmp(token, 'PRIMCOORD')
            atom_count_line = fgetl(fid);
            line_number = line_number + 1;
            if ~ischar(atom_count_line)
                error('WFOptimized:MalformedHeader', '%s ends after PRIMCOORD.', file_path);
            end
            atom_count = sscanf(atom_count_line, '%d', 1);
            if isempty(atom_count) || atom_count < 0
                error('WFOptimized:MalformedAtoms', '%s has an invalid atom count.', file_path);
            end
            atoms = cell(atom_count, 1);
            for atom_index = 1:atom_count
                atom_line = fgetl(fid);
                line_number = line_number + 1;
                if ~ischar(atom_line)
                    error('WFOptimized:MalformedAtoms', '%s ends inside PRIMCOORD.', file_path);
                end
                fields = strsplit(strtrim(atom_line));
                if numel(fields) < 4
                    error('WFOptimized:MalformedAtoms', '%s has an invalid atom record.', file_path);
                end
                coordinates = str2double(fields(2:4));
                if any(~isfinite(coordinates))
                    error('WFOptimized:MalformedAtoms', '%s has nonnumeric atom coordinates.', file_path);
                end
                atoms{atom_index} = struct('label', fields{1}, 'position', coordinates);
            end
            found_primcoord = true;
        elseif strcmp(token, 'BEGIN_DATAGRID_3D_UNKNOWN') || strcmp(token, 'BEGIN_DATAGRID_3D')
            dims_line = fgetl(fid);
            origin_line = fgetl(fid);
            vector_lines = cell(3, 1);
            line_number = line_number + 5;
            for k = 1:3
                vector_lines{k} = fgetl(fid);
            end
            if ~ischar(dims_line) || ~ischar(origin_line) || any(cellfun(@(x) ~ischar(x), vector_lines))
                error('WFOptimized:MalformedGrid', '%s ends inside DATAGRID geometry.', file_path);
            end
            dims = sscanf(dims_line, '%f', 3).';
            origin = sscanf(origin_line, '%f', 3).';
            for k = 1:3
                vectors(k, :) = sscanf(vector_lines{k}, '%f', 3).';
            end
            if numel(dims) ~= 3 || any(~isfinite(dims)) || any(dims <= 0) || any(dims ~= floor(dims))
                error('WFOptimized:MalformedGrid', '%s has invalid grid dimensions.', file_path);
            end
            if numel(origin) ~= 3 || any(~isfinite(origin)) || any(~isfinite(vectors(:)))
                error('WFOptimized:MalformedGrid', '%s has invalid grid geometry.', file_path);
            end
            dims = double(dims);
            data_start_line = line_number + 1;
            found_grid = true;
            break;
        end
    end

    if ~found_primcoord
        error('WFOptimized:MissingPrimcoord', '%s has no PRIMCOORD section.', file_path);
    end
    if ~found_grid
        error('WFOptimized:MissingGrid', '%s has no DATAGRID_3D section.', file_path);
    end
end

function wf_fast_close_file(fid)
    if fid >= 0
        fclose(fid);
    end
end
