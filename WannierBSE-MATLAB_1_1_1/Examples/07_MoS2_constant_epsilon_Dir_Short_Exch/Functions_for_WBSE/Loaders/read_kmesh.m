function [f, Nv_coarse, Nv_dense, N_co, N_de, k] = read_kmesh(file_path)
%==========================================================================
% READ_KMESH
%
% DESCRIPTION:
%   Reads WBSE k-mesh metadata and k-point coordinates from a generated text file.
%
% INPUT ARGUMENTS:
%   file_path : optional path to kmesh_WBSE-style input.
%
% OUTPUT:
%   f : coarse-to-dense range ratio.
%   Nv_coarse, Nv_dense : coarse and dense mesh parameters.
%   N_co, N_de : coarse and dense point counts.
%   k : N-by-2 in-plane k-point coordinates.
%==========================================================================

    if nargin < 1 || isempty(file_path)
        % Fallback to the new standard location if no path is provided
        file_path = 'User_input/kmesh.txt'; 
    end

    % Open the file
    fid = fopen(file_path, 'r');
    if fid == -1
        % Second fallback check: look in current folder if path fails
        fid = fopen('kmesh.txt', 'r');
        if fid == -1
            error('Cannot open file: %s', file_path);
        end
    end

    % Read parameters
    Dense_range = fscanf(fid, '%f', 1); % Line 1 is now Dense_range
    Nv_coarse   = fscanf(fid, '%f', 1); % Line 2
    Nv_dense    = fscanf(fid, '%f', 1); % Line 3
    f = Nv_coarse / Dense_range;
    N_co       = fscanf(fid, '%f', 1);
    N_de       = fscanf(fid, '%f', 1);

    % Read k-points (3 columns: kx, ky, kz), keep only kx, ky
    k_all = fscanf(fid, '%f', [3, inf]);
    k = k_all(1:2, :).'; % Extract kx and ky, transpose to [N x 2]

    % Close file
    fclose(fid);
end
