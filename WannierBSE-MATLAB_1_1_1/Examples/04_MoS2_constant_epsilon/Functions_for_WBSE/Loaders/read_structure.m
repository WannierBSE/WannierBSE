function [a1, a2, a3, a, tau, N_atoms] = read_structure(user_input_path)
%==========================================================================
% READ_STRUCTURE
%
% DESCRIPTION:
%   Loads lattice vectors and atomic coordinates from Wannier90 or structure input files.
%
% INPUT ARGUMENTS:
%   user_input_path : optional User_input directory path.
%
% OUTPUT:
%   a1, a2, a3 : real-space lattice vectors.
%   a : in-plane lattice constant norm.
%   tau : atomic Cartesian coordinates.
%   N_atoms : number of atoms in the unit cell.
%==========================================================================

persistent structure_source_reported

if nargin == 0
    loader_path = fileparts(mfilename('fullpath'));
    project_root = fullfile(loader_path, '..', '..');
    user_input_path = fullfile(project_root, 'User_input');
end

wannier90_path = fullfile(user_input_path, 'wannier90.win');
structure_path = fullfile(user_input_path, 'structure.txt');

if isfile(wannier90_path)
    source_path = wannier90_path;
    source_name = fullfile('User_input', 'wannier90.win');
elseif isfile(structure_path)
    source_path = structure_path;
    source_name = fullfile('User_input', 'structure.txt');
else
    error('WannierBSE:StructureNotFound', ...
        'No structure input was found. Expected either %s or %s.', ...
        wannier90_path, structure_path);
end

fid = fopen(source_path, 'r');
if fid == -1
    error('WannierBSE:StructureOpenFailed', ...
        'Cannot open structure input: %s', source_path);
end
cleanup_obj = onCleanup(@() fclose(fid));

lines = strings(0, 1);
while ~feof(fid)
    lines(end + 1, 1) = string(fgetl(fid)); %#ok<AGROW>
end

lattice_begin = find(lower(strtrim(lines)) == "begin unit_cell_cart", 1);
lattice_end = find(lower(strtrim(lines)) == "end unit_cell_cart", 1);
atoms_cart_begin = find(lower(strtrim(lines)) == "begin atoms_cart", 1);
atoms_frac_begin = find(lower(strtrim(lines)) == "begin atoms_frac", 1);

if isempty(atoms_cart_begin) || (~isempty(atoms_frac_begin) && atoms_frac_begin < atoms_cart_begin)
    atoms_begin = atoms_frac_begin;
    atom_block_type = "frac";
else
    atoms_begin = atoms_cart_begin;
    atom_block_type = "cart";
end
if atom_block_type == "frac"
    atoms_end = find(lower(strtrim(lines)) == "end atoms_frac", 1);
else
    atoms_end = find(lower(strtrim(lines)) == "end atoms_cart", 1);
end

if isempty(lattice_begin) || isempty(lattice_end) || lattice_end <= lattice_begin
    error('WannierBSE:InvalidStructure', ...
        'Missing or incomplete unit_cell_cart block in %s.', source_path);
end
if isempty(atoms_begin) || isempty(atoms_end) || atoms_end <= atoms_begin + 1
    error('WannierBSE:InvalidStructure', ...
        'Missing or empty atoms_cart block in %s.', source_path);
end

lattice_vectors = zeros(3, 3);
lattice_count = 0;
for line_index = lattice_begin + 1:lattice_end - 1
    line_text = strtrim(char(lines(line_index)));
    if isempty(line_text)
        continue
    end
    if strcmpi(line_text, 'ang')
        continue
    elseif strcmpi(line_text, 'bohr')
        error('WannierBSE:UnsupportedStructureUnits', ...
            'Unsupported unit declaration ''bohr'' in %s.', source_path);
    end
    lattice_count = lattice_count + 1;
    if lattice_count > 3
        error('WannierBSE:InvalidStructure', ...
            'More than three lattice vectors found in %s.', source_path);
    end
    lattice_vectors(lattice_count, :) = parse_vector(lines(line_index), source_path);
end
if lattice_count ~= 3
    error('WannierBSE:InvalidStructure', ...
        'Missing or incomplete unit_cell_cart block in %s.', source_path);
end
a1 = lattice_vectors(1, :);
a2 = lattice_vectors(2, :);
a3 = lattice_vectors(3, :);
a = norm(a1);

N_atoms = atoms_end - atoms_begin - 1;
tau = zeros(N_atoms, 3);
for atom_index = 1:N_atoms
    atom_values = sscanf(char(lines(atoms_begin + atom_index)), '%*s %f %f %f');
    if numel(atom_values) ~= 3
        error('WannierBSE:InvalidStructure', ...
            'Invalid atomic coordinate in %s at atom %d.', ...
            source_path, atom_index);
    end
    tau(atom_index, :) = atom_values.';
end
if atom_block_type == "frac"
    tau = tau * [a1; a2; a3];
end

if isempty(structure_source_reported)
    fprintf('[INFO ] Structure extracted from %s.\n', source_name);
    structure_source_reported = true;
end

clear cleanup_obj
end


function vector = parse_vector(line, source_path)
%==========================================================================
% PARSE_VECTOR
%
% DESCRIPTION:
%   Parses one lattice-vector line from a structure file.
%
% OUTPUT:
%   vector : three-component row vector.
%==========================================================================

tokens = regexp(strtrim(char(line)), '\s+', 'split');
vector = sscanf(char(line), '%f');
if numel(tokens) ~= 3 || numel(vector) ~= 3
    error('WannierBSE:InvalidStructure', ...
        'Invalid lattice vector in %s.', source_path);
end
vector = vector.';
end
