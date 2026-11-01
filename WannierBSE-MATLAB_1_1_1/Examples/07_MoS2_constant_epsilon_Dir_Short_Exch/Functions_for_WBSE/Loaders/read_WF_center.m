function varargout = read_WF_center(varargin)
%==========================================================================
% READ_WF_CENTER
%
% DESCRIPTION:
%   Reads Wannier-center metadata and optionally converts it to the legacy BSE atom/spinor index layout.
%
% INPUT ARGUMENTS:
%   varargin : filename, N_atoms, or N_atoms plus filename.
%
% OUTPUT:
%   varargout : metadata structure or legacy WF_center and Basis_tau_index_cell outputs.
%==========================================================================

if nargin == 0
    filename = default_wf_centers_path();
    mode = "struct";
elseif nargin == 1
    if isnumeric(varargin{1})
        N_atoms = varargin{1};
        filename = default_wf_centers_path();
        mode = "legacy";
    else
        filename = varargin{1};
        mode = "struct";
    end
elseif nargin == 2
    N_atoms = varargin{1};
    filename = varargin{2};
    mode = "legacy";
else
    error('read_WF_center:InvalidInput', 'Expected filename or N_atoms, filename.');
end

wf = parse_wf_centers(filename);

if mode == "struct"
    if nargout > 1
        error('read_WF_center:InvalidOutput', ...
              'Path-only usage returns one struct output.');
    end
    varargout{1} = wf;
    return;
end

if ~isscalar(N_atoms) || ~isfinite(N_atoms) || N_atoms < 1 || fix(N_atoms) ~= N_atoms
    error('read_WF_center:InvalidAtomCount', 'N_atoms must be a positive integer scalar.');
end

[WF_center, Basis_tau_index_cell] = wf_to_legacy_outputs(wf, N_atoms);
varargout{1} = WF_center;
if nargout > 1
    varargout{2} = Basis_tau_index_cell;
end
end

function filename = default_wf_centers_path()
%==========================================================================
% DEFAULT_WF_CENTERS_PATH
%
% DESCRIPTION:
%   Selects the default WF_centers.txt path from the base workspace or legacy parameter location.
%
% OUTPUT:
%   filename : resolved default WF centers path.
%==========================================================================

    if evalin('base', 'exist(''ParamPath'', ''var'')')
        ParamPath = evalin('base', 'ParamPath');
        filename = fullfile(ParamPath, 'WF_centers.txt');
    else
        filename = '../parameters/WF_centers.txt';
    end
end

function wf = parse_wf_centers(filename)
%==========================================================================
% PARSE_WF_CENTERS
%
% DESCRIPTION:
%   Parses WF center indices, atom assignments, and spinor labels from text input.
%
% OUTPUT:
%   wf : normalized WF-center metadata structure.
%==========================================================================

    if ~isfile(filename)
        error('read_WF_center:FileNotFound', 'WF centers file not found: %s', filename);
    end

    raw_lines = readlines(filename);
    raw_lines = raw_lines(strlength(strtrim(raw_lines)) > 0);
    if numel(raw_lines) < 2
        error('read_WF_center:InvalidFormat', 'Invalid WF centers file: %s', filename);
    end

    header_idx = find_wf_header(raw_lines);
    if isempty(header_idx) || header_idx == numel(raw_lines)
        error('read_WF_center:InvalidFormat', ...
              'Missing WF_centers table header or data rows in %s.', filename);
    end

    description_lines = raw_lines(1:header_idx - 1);
    if isempty(description_lines)
        description = '';
    else
        description = char(strjoin(description_lines, newline));
    end

    header_tokens = split(strtrim(raw_lines(header_idx)));
    has_spinor = any(strcmpi(header_tokens, 'Spinor'));
    data_lines = raw_lines(header_idx + 1:end);
    NB = numel(data_lines);

    index = zeros(NB, 1);
    atom = zeros(NB, 1);
    spinor = strings(NB, 1);
    for r = 1:NB
        source_line = header_idx + r;
        tok = split(strtrim(data_lines(r)));
        if numel(tok) < 2
            error('read_WF_center:InvalidFormat', ...
                  'Row %d has fewer than two columns in %s.', source_line, filename);
        end
        index(r) = str2double(tok(1));
        atom(r) = str2double(tok(2));
        if ~isfinite(index(r)) || ~isfinite(atom(r))
            error('read_WF_center:InvalidFormat', ...
                  'Row %d has invalid numeric fields in %s.', source_line, filename);
        end
        if has_spinor
            if numel(tok) < 3
                error('read_WF_center:MissingSpinor', ...
                      'Spinor column declared but row %d has no label.', source_line);
            end
            spinor(r) = normalize_spinor_label(tok(3));
        else
            if mod(index(r), 2) == 1
                spinor(r) = "up";
            else
                spinor(r) = "down";
            end
        end
    end

    if any(index(:) ~= (1:NB).')
        error('read_WF_center:InvalidIndex', ...
              'WF indices in %s must be contiguous 1:NB.', filename);
    end

    wf = struct('description', description, ...
                'index', index, ...
                'atom', atom, ...
                'spinor', spinor, ...
                'header_line', header_idx);
end

function header_idx = find_wf_header(lines)
%==========================================================================
% FIND_WF_HEADER
%
% DESCRIPTION:
%   Locates the WF_center table header in a text file.
%
% OUTPUT:
%   header_idx : one-based header line index, or empty if absent.
%==========================================================================

    header_idx = [];
    for n = 1:numel(lines)
        tok = split(strtrim(lines(n)));
        if numel(tok) >= 2 && strcmpi(tok(1), 'i') && strcmpi(tok(2), 'I')
            header_idx = n;
            return;
        end
    end
end

function spinor = normalize_spinor_label(label)
%==========================================================================
% NORMALIZE_SPINOR_LABEL
%
% DESCRIPTION:
%   Normalizes accepted spinor label variants to up or down.
%
% OUTPUT:
%   spinor : canonical spinor string.
%==========================================================================

    s = lower(strtrim(string(label)));
    if any(s == ["up", "u", "+", "+1", "1"])
        spinor = "up";
    elseif any(s == ["down", "dn", "d", "-", "-1", "2"])
        spinor = "down";
    else
        error('read_WF_center:InvalidSpinor', 'Invalid Spinor label "%s".', label);
    end
end

function [WF_center, Basis_tau_index_cell] = wf_to_legacy_outputs(wf, N_atoms)
%==========================================================================
% WF_TO_LEGACY_OUTPUTS
%
% DESCRIPTION:
%   Converts parsed WF metadata to legacy arrays consumed by direct and exchange Coulomb routines.
%
% OUTPUT:
%   WF_center : NB-by-3 center table.
%   Basis_tau_index_cell : atom-indexed cell array of Wannier basis indices.
%==========================================================================

    NB = numel(wf.index);
    WF_center = zeros(NB, 3);
    WF_center(:, 1) = wf.index;
    WF_center(:, 2) = wf.atom;
    WF_center(wf.spinor == "up", 3) = 1;
    WF_center(wf.spinor == "down", 3) = 2;

    Basis_tau_index_cell = cell(N_atoms, 1);
    for Ni = 1:N_atoms
        Basis_tau_index_cell{Ni} = find(WF_center(:, 2) == Ni);
    end
end
