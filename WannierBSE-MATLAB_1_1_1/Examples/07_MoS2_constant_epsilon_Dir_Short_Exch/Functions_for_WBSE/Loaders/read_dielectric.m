function [q_data, epsQ2D_data] = read_dielectric(file_path)
%==========================================================================
% READ_DIELECTRIC
%
% DESCRIPTION:
%   Loads the q-dependent effective dielectric table used by the screened direct Coulomb kernel.
%
% INPUT ARGUMENTS:
%   file_path : optional path to epsilon.txt.
%
% OUTPUT:
%   q_data : q-grid column.
%   epsQ2D_data : effective dielectric values.
%==========================================================================

    if nargin < 1 || isempty(file_path)
        % Fallback to the new standard location
        file_path = 'User_input/epsilon.txt'; 
    end

    % Check if file exists
    if ~isfile(file_path)
        % Second fallback: check the local folder
        if isfile('epsilon.txt')
            file_path = 'epsilon.txt';
        else
            error('Read_dielectric: File not found at %s', file_path);
        end
    end
    
    data = importdata(file_path);
    
    % Extract the first and second columns
    q_data = data(:, 1);
    epsQ2D_data = data(:, 2);
end
