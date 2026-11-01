function [W_up_t,W_down_t,tin_all,r_t,u_t,v_t,z_t,u_point_t,v_point_t,z_point_t,Nu_t,Nv_t,Nz_t] = find_wfs_box( ...
    Tol_probability, W_up, W_down, r, u, v, z, deltaV, Na1, z_point)
%==========================================================================
% FIND_WFS_BOX
%
% DESCRIPTION:
%   Finds and extracts the minimal shared real-space grid box needed to retain localized Wannier probability density.
%
% INPUT ARGUMENTS:
%   Tol_probability : retained Wannier probability threshold.
%   W_up, W_down : spinor Wannier functions.
%   r, u, v, z : real-space grid coordinates.
%   deltaV : real-space volume element.
%   Na1 : first in-plane grid dimension.
%   z_point : z-grid coordinates.
%
% OUTPUT:
%   W_up_t, W_down_t : Wannier functions restricted to the selected box.
%==========================================================================




%===================== find truncated box =========================
uv_Box = zeros(6,size(W_up,2));

for i=1:size(W_up,2)

%-----------normalization by Riemann sum--------------
    Wi = (conj(W_up(:,i)).*W_up(:,i))+(conj(W_down(:,i)).*W_down(:,i));
    N = deltaV.*sum(Wi);

    W_up_Riemann = W_up(:,i)./sqrt(N);
    W_down_Riemann = W_down(:,i)./sqrt(N);

%-----------normalization by Riemann sum--------------

%------------------sort W-------------------
W_abs = (conj(W_up_Riemann).*W_up_Riemann)+(conj(W_down_Riemann).*W_down_Riemann);
[W_sorted,ind] = sort(W_abs,'descend');
r_sorted = r(ind,:);
u_sorted = u(ind);
v_sorted = v(ind);
z_sorted = z(ind);

sumW = deltaV.*cumsum(W_sorted);
n = find(sumW<=Tol_probability);
n = max(n);

uv_Box(1,i) = min(u_sorted(1:n));
uv_Box(2,i) = max(u_sorted(1:n));
uv_Box(3,i) = min(v_sorted(1:n));
uv_Box(4,i) = max(v_sorted(1:n));
uv_Box(5,i) = min(z_sorted(1:n));
uv_Box(6,i) = max(z_sorted(1:n));
%------------------sort W-------------------

end




uv_Box_BSE(1,1) = (min(uv_Box(1,:)));
uv_Box_BSE(2,1) = (max(uv_Box(2,:)));
uv_Box_BSE(3,1) = (min(uv_Box(3,:)));
uv_Box_BSE(4,1) = (max(uv_Box(4,:)));
uv_Box_BSE(5,1) = (min(uv_Box(5,:)));
uv_Box_BSE(6,1) = (max(uv_Box(6,:)));

u_Min = uv_Box_BSE(1,1);
u_Max = uv_Box_BSE(2,1);
v_Min = uv_Box_BSE(3,1);
v_Max = uv_Box_BSE(4,1);
z_Min = uv_Box_BSE(5,1);
z_Max = uv_Box_BSE(6,1);

uv_edge = [[u_Min-0.01,v_Min-0.01];...
           [u_Max+0.01,v_Min-0.01];...
           [u_Max+0.01,v_Max+0.01];...
           [u_Min-0.01,v_Max+0.01]];


[in,on] = inpolygon(u(1:(Na1*Na1)),v(1:(Na1*Na1)),uv_edge(:,1),uv_edge(:,2));
tin = find(in==1);


ind_zmin = find(abs(z_point-z_Min)<(1e-6));
ind_zmax = find(abs(z_point-z_Max)<(1e-6));
ind_z = ind_zmin:ind_zmax;


tin_all = zeros(length(tin)*length(ind_z),1);
r_t = zeros(length(tin)*length(ind_z),3);
u_t = zeros(length(tin)*length(ind_z),1);
v_t = zeros(length(tin)*length(ind_z),1);
z_t = zeros(length(tin)*length(ind_z),1);
for iz=1:length(ind_z)
tin_all((1:length(tin))+((iz-1)*length(tin))) = tin+((ind_z(iz)-1)*Na1*Na1);
r_t((1:length(tin))+((iz-1)*length(tin)),:) = r(tin+((ind_z(iz)-1)*Na1*Na1),:);
u_t((1:length(tin))+((iz-1)*length(tin)),:) = u(tin+((ind_z(iz)-1)*Na1*Na1),:);
v_t((1:length(tin))+((iz-1)*length(tin)),:) = v(tin+((ind_z(iz)-1)*Na1*Na1),:);
z_t((1:length(tin))+((iz-1)*length(tin)),:) = z(tin+((ind_z(iz)-1)*Na1*Na1),:);
end


W_up_t = W_up(tin_all,:);
W_down_t = W_down(tin_all,:);

u_point_t = uniquetol(u_t,1e-3);
v_point_t = uniquetol(v_t,1e-3);
z_point_t = uniquetol(z_t,1e-3);


Nu_t = length(u_point_t);
Nv_t = length(v_point_t);
Nz_t = length(z_point_t);

%===================== find truncated box =========================




end
