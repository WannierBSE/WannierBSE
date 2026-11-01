function [W_up, W_down, r, deltaV, Na1, Na2, Nz] = Read_WFs_spinor(NB)
%==========================================================================
% READ_WFS_SPINOR
%
% DESCRIPTION:
%   Assembles complex spinor Wannier functions from real and imaginary component files produced by the
%   Wannier engine.
%
% INPUT ARGUMENTS:
%   NB : number of Wannier functions.
%
% OUTPUT:
%   W_up, W_down : complex spinor components.
%   r : Cartesian grid coordinates.
%   deltaV : real-space volume element.
%   Na1, Na2, Nz : grid dimensions.
%==========================================================================

GridInfo = importdata('GridInfo.txt');
Na1 = GridInfo(1,1);
Na2 = GridInfo(1,2);
Nz  = GridInfo(1,3);
a1  = GridInfo(3,:);
a2  = GridInfo(4,:);
a3  = GridInfo(5,:);

% Calculate differential volume element
deltaV = (dot(cross(a1,a2),a3)) / ((Na1-1)*(Na2-1)*(Nz-1));

% Pre-allocate spinor arrays
total_points = Na1 * Na2 * Nz;
W_up   = zeros(total_points, NB);
W_down = zeros(total_points, NB);

% --- Data Assembly Loop ---
spin_types = {'up', 'down'};

for s = 1:2
    spin_Name = spin_types{s};
    
    for i = 1:NB
        % Formatting index with leading zero (e.g., 01, 02... 10, 11)
        if i < 10
            idx_str = ['0', num2str(i)];
        else
            idx_str = num2str(i);
        end
        
        % --- 1. Load Real Part ---
        folder_real = [spin_Name, '_real'];
        file_real_base = ['WDATA_', spin_Name, '_real_', idx_str];
        full_path_real = fullfile('Output', folder_real, [file_real_base, '.mat']);
        
        S_real = load(full_path_real);
        WF_real = S_real.(file_real_base);
        
        % --- 2. Load Imaginary Part ---
        folder_imag = [spin_Name, '_imag'];
        file_imag_base = ['WDATA_', spin_Name, '_imag_', idx_str];
        full_path_imag = fullfile('Output', folder_imag, [file_imag_base, '.mat']);
        
        S_imag = load(full_path_imag);
        WF_imag = S_imag.(file_imag_base);
        
        % --- 3. Construct Complex Spinor ---
        % Column 4 contains the wavefunction amplitude values
        if s == 1
            W_up(:,i)   = WF_real(:,4) + 1i*WF_imag(:,4);
        else
            W_down(:,i) = WF_real(:,4) + 1i*WF_imag(:,4);
        end
    end
end

% --- Extract Grid Coordinates ---
% Pulling the r-grid (first 3 columns) from the first available real data file
sample_file = fullfile('Output', 'up_real', 'WDATA_up_real_01.mat');
S_sample = load(sample_file);
r = S_sample.WDATA_up_real_01(:, 1:3);

end
