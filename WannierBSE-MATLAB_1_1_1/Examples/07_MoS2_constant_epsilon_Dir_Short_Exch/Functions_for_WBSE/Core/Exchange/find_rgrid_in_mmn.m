function [R_mNN,n1_n2_mNN,d_max] = find_rgrid_in_mmn( ...
    a1, a2, tau, N_atoms, supercell_size, Mmax, NN_distance_tolerance, Mmax_tolerance)
%==========================================================================
% FIND_RGRID_IN_MMN
%
% DESCRIPTION:
%   Enumerates lattice translations included by the m-th-nearest-neighbor Wannier cutoff for exchange calculations.
%
% INPUT ARGUMENTS:
%   a1, a2, tau : lattice vectors and atomic positions.
%   N_atoms, supercell_size : geometry and translation-grid dimensions.
%   Mmax, NN_distance_tolerance, Mmax_tolerance : exchange cutoff controls.
%
% OUTPUT:
%   R_mNN : retained lattice translation vectors.
%   n1_n2_mNN : integer lattice indices for retained translations.
%   d_max : cutoff distance associated with Mmax.
%==========================================================================



inplane_index = [1,2];

n1n2_grid = zeros(supercell_size^2,2);
R_grid = zeros(supercell_size^2,3);
for j=1:supercell_size
for i=1:supercell_size
    
if mod(supercell_size,2)~=0
n1n2_grid(j+((i-1)*supercell_size),:)= [((i-1)-((supercell_size-1)/2)),((j-1)-((supercell_size-1)/2))];
R_grid(j+((i-1)*supercell_size),:) = (((i-1)-((supercell_size-1)/2)).*a1)...
                                  +(((j-1)-((supercell_size-1)/2)).*a2);
elseif mod(supercell_size,2)==0
n1n2_grid(j+((i-1)*supercell_size),:)= [((i-1)-((supercell_size)/2)),((j-1)-((supercell_size)/2))];
R_grid(j+((i-1)*supercell_size),:) = (((i-1)-((supercell_size)/2)).*a1)...
                                  +(((j-1)-((supercell_size)/2)).*a2);
end

end
end

%-------------------------
for i=1:size(R_grid,1)
    for j=1:N_atoms
    atom_position(j+(i-1)*N_atoms,:) = (tau(j,:)+R_grid(i,:));
    R_for_atoms(j+(i-1)*N_atoms,:) = R_grid(i,:);
    n1n2_grid_for_atoms(j+(i-1)*N_atoms,:) = n1n2_grid(i,:);
    end
end

for i=1:N_atoms
    distance_inplane_mNN(:,i) = sqrt(sum((atom_position(:,inplane_index)-tau(i,inplane_index)).^2,2));
    [sorted_dm_all(:,i),sort_ind(:,i)] = sort(distance_inplane_mNN(:,i));

    distance_info(i).diff_tau_R = sorted_dm_all(:,i);
    distance_info(i).Rgrid = R_for_atoms(sort_ind(:,i),:);
    distance_info(i).n1n2grid = n1n2_grid_for_atoms(sort_ind(:,i),:);
end

%------------find dm (distance of m-th neighbor atom)-----------
for i=1:N_atoms
    dm_all_i = sorted_dm_all(:,i);

    for j=1:length(dm_all_i)
    Diff_dm_all_i = dm_all_i-dm_all_i(j);
    index_of_same_dm = find(abs(Diff_dm_all_i)<=NN_distance_tolerance);
    if numel(index_of_same_dm)>=2
        dm_NN_i(j,1) = max(dm_all_i(index_of_same_dm)); % remove the duplicate data
    end
    end
    dm_each_atom(i).data = uniquetol(dm_NN_i,1e-10);

end

%-----summarize the m-th NN distance in each atom
L = 0;
for i=1:N_atoms
Sn = size(dm_each_atom(i).data,1);
dm_temp((1:Sn)+L,:) = dm_each_atom(i).data;
L = size(dm_temp,1);
end
dm_temp = sort(dm_temp,'ascend');

for j=1:length(dm_temp)
Diff_dm = dm_temp-dm_temp(j);
index_of_same_dm = find(abs(Diff_dm)<=NN_distance_tolerance);
if numel(index_of_same_dm)>=2
    dm(j,1) = max(dm_temp(index_of_same_dm));
end
end

dm = uniquetol(dm,1e-10);
d_max = dm(1+Mmax);
%-----summarize the m-th NN distance in each atom

%------------find dm (distance of m-th neighbor atom)-----------

%------------find the R for BSE calculation---------
for i=1:N_atoms
diff_tau_R = distance_info(i).diff_tau_R;
t_max = find(diff_tau_R<=(d_max+Mmax_tolerance));
mNN_info(i).n1n2 = distance_info(i).n1n2grid(t_max,:);
mNN_info(i).Rgrid = distance_info(i).Rgrid(t_max,:);
end

L = 0;
for i=1:N_atoms
Sn = size(mNN_info(i).n1n2,1);
n1n2_temp((1:Sn)+L,:) = mNN_info(i).n1n2;
R_temp((1:Sn)+L,:) = mNN_info(i).Rgrid;
L = size(n1n2_temp,1);
end

[n1_n2_mNN,IA,IC] = uniquetol(n1n2_temp,1e-4,'ByRows',true);
R_mNN = R_temp(IA,:);

R_check = ( (n1_n2_mNN(:,1).*a1)+(n1_n2_mNN(:,2).*a2) );
Diff_R = R_mNN-R_check;
%------------find the R for BSE calculation---------


end
