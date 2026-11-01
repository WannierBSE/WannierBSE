function mismatch = check_cached_kmesh_config(cache_file, control_file)
%CHECK_CACHED_KMESH_CONFIG Compare reliable cached-header mesh controls.
%
% The cache header is written in the order
%   Dense_range, Nk_coarse, Nk_dense, total_coarse, total_dense.
% Only the first three fields correspond to kmesh_control.txt controls.
% Invalid or incomplete inputs are treated as not comparable.

    mismatch = false;

    if ~isfile(cache_file) || ~isfile(control_file)
        return;
    end

    try
        params = read_params(control_file);
        required_names = {'Nk_coarse', 'Nk_dense', 'Dense_range'};
        control_values = zeros(1, 3);
        for i = 1:numel(required_names)
            name = required_names{i};
            if ~isfield(params, name) || ~isnumeric(params.(name)) || ...
                    ~isscalar(params.(name)) || ~isfinite(params.(name)) || ...
                    params.(name) ~= floor(params.(name))
                return;
            end
            control_values(i) = params.(name);
        end

        fid = fopen(cache_file, 'r');
        if fid < 0
            return;
        end
        cleanup_obj = onCleanup(@() fclose(fid));
        header = zeros(5, 1);
        for i = 1:5
            line = fgetl(fid);
            if ~ischar(line)
                return;
            end
            header(i) = str2double(strtrim(line));
        end
        clear cleanup_obj;

        % Require all five header fields so truncated cache metadata is not
        % interpreted as a confirmed mismatch.
        if numel(header) < 5 || any(~isfinite(header)) || any(header ~= floor(header))
            return;
        end

        cached_values = [header(2), header(3), header(1)];
        mismatch = any(cached_values ~= control_values);
    catch
        mismatch = false;
    end
end
