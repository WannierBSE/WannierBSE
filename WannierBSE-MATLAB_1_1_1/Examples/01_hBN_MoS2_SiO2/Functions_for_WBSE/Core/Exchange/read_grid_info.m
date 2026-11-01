function [Na1,Na2,Na3,deltaV,supercell_size,shift_a1,shift_a2,shift_a3] = read_grid_info(WFPath,a1)
%==========================================================================
% READ_GRID_INFO
%
% DESCRIPTION:
%   Reads Wannier grid dimensions, volume element, and supercell metadata from XSF-derived grid files.
%
% INPUT ARGUMENTS:
%   WFPath : Wannier-function data path.
%   a1 : lattice vector used for supercell scaling.
%
% OUTPUT:
%   Na1, Na2, Na3 : grid dimensions.
%   deltaV : real-space volume element.
%   supercell_size : inferred in-plane supercell size.
%   shift_a1, shift_a2, shift_a3 : grid shifts.
%==========================================================================


GridInfo = importdata(fullfile(WFPath, 'GridInfo.txt'));
Na1 = GridInfo(1,1);
Na2 = GridInfo(1,2);
Na3 = GridInfo(1,3);
Nz = Na3;
A1s = GridInfo(3,:);
A2s = GridInfo(4,:);
A3s = GridInfo(5,:);
deltaV = (dot(cross(A1s,A2s),A3s))/((Na1-1)*(Na2-1)*(Na3-1));

supercell_size = ceil(norm(A1s)./(norm(a1)));

shift_a1 = Na1/supercell_size;
shift_a2 = Na2/supercell_size;
shift_a3 = 0;

end
