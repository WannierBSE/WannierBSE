function params = read_params(fname)
%==========================================================================
% READ_PARAMS
%
% DESCRIPTION:
%   Reads key-value control parameters into a MATLAB structure while preserving numeric vectors and
%   string values.
%
% INPUT ARGUMENTS:
%   fname : parameter-file path.
%
% OUTPUT:
%   params : structure whose field names are normalized from parameter keys.
%==========================================================================

    fid = fopen(fname, 'r');
    if fid < 0
        error('Cannot open file: %s', fname);
    end

    params = struct();

    while ~feof(fid)
        line = strtrim(fgetl(fid));
        % Skip empty or comment-only lines
        if isempty(line) || startsWith(line, '%')
            continue;
        end

        % Remove inline comment
        commentIdx = strfind(line, '%');
        if ~isempty(commentIdx)
            line = strtrim(line(1:commentIdx(1)-1));
        end

        % Match key = value
        tokens = regexp(line, '^\s*([A-Za-z_][A-Za-z0-9_ ]*)\s*=\s*(.+)$', 'tokens');
        if isempty(tokens)
            continue;
        end

        key = regexprep(strtrim(tokens{1}{1}), '\s+', '_');
        val_str = strtrim(tokens{1}{2});

        % Attempt to parse numeric value or vector
        num_vals = str2num(val_str); %#ok<ST2NM>
        if ~isempty(num_vals)
            params.(key) = num_vals;
        else
            % Store as string if not numeric
            params.(key) = val_str;
        end
    end

    fclose(fid);
