function value = parse_optional_boolean(raw_value, parameter_name, source_file)
%==========================================================================
% PARSE_OPTIONAL_BOOLEAN
%
% DESCRIPTION:
%   Strictly parses an optional Boolean setting. Accepted values are
%   case-insensitive true/false strings and numeric or logical 1/0 values.
%   Missing values are handled by the caller; this helper validates only
%   values that are present.
%
% INPUT ARGUMENTS:
%   raw_value : logical, numeric, character, or string Boolean value.
%   parameter_name : parameter name used in validation messages.
%   source_file : source-file name used in validation messages.
%
% OUTPUT:
%   value : validated logical scalar.
%==========================================================================

    if isempty(raw_value)
        error('WBSE:InvalidBooleanParameter', ...
              'Parameter %s in %s must be true or false; an empty value is invalid.', ...
              parameter_name, source_file);
    end

    if islogical(raw_value) && isscalar(raw_value)
        value = raw_value;
        return;
    end

    if isnumeric(raw_value) && isscalar(raw_value) && isfinite(raw_value) && (raw_value == 0 || raw_value == 1)
        value = logical(raw_value);
        return;
    end

    if isstring(raw_value) && isscalar(raw_value)
        text_value = strtrim(char(raw_value));
    elseif ischar(raw_value) && isrow(raw_value)
        text_value = strtrim(raw_value);
    else
        text_value = '';
    end

    if strcmpi(text_value, 'true')
        value = true;
    elseif strcmpi(text_value, 'false')
        value = false;
    else
        if isempty(text_value)
            text_value = mat2str(raw_value);
        end
        error('WBSE:InvalidBooleanParameter', ...
              'Parameter %s in %s must be true or false (optionally 1 or 0); received %s.', ...
              parameter_name, source_file, text_value);
    end
end
