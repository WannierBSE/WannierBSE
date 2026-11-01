function [W_up_t,W_down_t,tin_all,r_t,u_t,v_t,z_t,u_point_t,v_point_t,z_point_t,Nu_t,Nv_t,Nz_t,Na1_t,Na2_t,zcut_min,zcut_max] = ...
                                               find_truncated_wfs(W_up,W_down,r,u,v,deltaV,Na1,z_point,Tol_probability)
%==========================================================================
% FIND_TRUNCATED_WFS
%
% DESCRIPTION:
%   Constructs a common truncated real-space box that retains the requested Wannier-function probability mass for exchange calculations.
%
% INPUT ARGUMENTS:
%   W_up, W_down : spinor Wannier functions.
%   r, u, v, deltaV : real-space grid coordinates and volume element.
%   Na1, z_point : in-plane grid size and z-grid coordinates.
%   Tol_probability : retained probability threshold.
%
% OUTPUT:
%   W_up_t, W_down_t : truncated spinor Wannier functions.
%   tin_all, r_t, u_t, v_t, z_t : retained grid indices and coordinates.
%   u_point_t, v_point_t, z_point_t : truncated spline grids.
%   Nu_t, Nv_t, Nz_t, Na1_t, Na2_t : truncated grid dimensions.
%   zcut_min, zcut_max : z-index bounds retained by the truncation.
%==========================================================================

[W_up_t,W_down_t,tin_all,r_t,u_t,v_t,z_t,u_point_t,v_point_t,z_point_t,Nu_t,Nv_t,Nz_t] ...
    = find_wfs_box(Tol_probability,W_up,W_down,r,u,v,r(:,3),deltaV,Na1,z_point);

Na1_t = length(u_point_t);
Na2_t = length(v_point_t);
zcut_min = 1;
zcut_max = Nz_t;
%================== find truncated box ====================
%------------------------Read Wannier functions----------------------------


end
