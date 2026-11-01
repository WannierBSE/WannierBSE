function [c3,c2,c1,c0] = wf_spline_interpolation(Wij_z,kex,a1,a2,u_2D,u_point_t,v_point_t,kexTol)
%==========================================================================
% WF_SPLINE_INTERPOLATION
%
% DESCRIPTION:
%   Builds cubic-spline coefficients for products of shifted Wannier functions on the truncated grid.
%
% INPUT ARGUMENTS:
%   Wij_z : z-dependent Wannier-pair product.
%   kex, a1, a2 : exchange momentum and in-plane lattice vectors.
%   u_2D, u_point_t, v_point_t : in-plane grids.
%   kexTol : momentum-coordinate tolerance.
%
% OUTPUT:
%   c3, c2, c1, c0 : spline coefficient tensors.
%==========================================================================


Nu = length(u_point_t);
Nv = length(v_point_t);

a1x = a1(1);
a1y = a1(2);
a2x = a2(1);
a2y = a2(2);
Beta = (-a2x*a1y)+(a1x*a2y);

kex_uu = ((kex(1)*a1x*a2y)+(kex(2)*a1y*a2y))./Beta;
kex_vv = ((kex(1)*a2x*a1y)+(kex(2)*a1y*a2y))./Beta;


if ( norm(kex_uu)<kexTol && norm(kex_vv)>kexTol )
kexr_u = sum(bsxfun(@times,u_2D,kex_uu),2);
kexr_v = zeros(size(v_point_t,1),1);
elseif ( norm(kex_uu)>kexTol && norm(kex_vv)<kexTol )
kexr_u = zeros(size(u_2D,1),1);
kexr_v = sum(bsxfun(@times,v_point_t,kex_vv),2);
elseif ( norm(kex_uu)<kexTol && norm(kex_vv)<kexTol )
kexr_u = sum(bsxfun(@times,u_2D,kex_uu),2);
kexr_v = sum(bsxfun(@times,v_point_t,kex_vv),2);
else
kexr_u = zeros(size(u_2D,1),1);
kexr_v = zeros(size(v_point_t,1),1);
end

%-------------c1 c2 c3 c4------------
% size of c1 = zeros(Nu-1,Nv,Nz);
% size of c2 = zeros(Nu-1,Nv,Nz);
% size of c3 = zeros(Nu-1,Nv,Nz);
% size of c4 = zeros(Nu-1,Nv,Nz);

% Wij = (conj(Wi_up).*Wj_up)+(conj(Wi_down).*Wj_down);
RW = reshape(Wij_z.*exp(-1i*kexr_u),[Nu,Nv]);

RW = permute(RW,[2,1]);
SPu = spline(u_point_t,RW);
c3 = SPu.coefs(:,1);
c2 = SPu.coefs(:,2);
c1 = SPu.coefs(:,3);
c0 = SPu.coefs(:,4);

c3 = reshape(c3,[Nv,Nu-1]);
c3 = permute(c3,[2,1]);

c2 = reshape(c2,[Nv,Nu-1]);
c2 = permute(c2,[2,1]);

c1 = reshape(c1,[Nv,Nu-1]);
c1 = permute(c1,[2,1]);

c0 = reshape(c0,[Nv,Nu-1]);
c0 = permute(c0,[2,1]);
%-------------c1 c2 c3 c4------------


%%%%%---------------this block for checking c1~c4 ---------------
% tic
% RW = reshape(Wij_z.*exp(-1i*kexr_u),[Nu,Nv]);
% c1_check = zeros(Nu-1,Nv);
% c2_check = zeros(Nu-1,Nv);
% c3_check = zeros(Nu-1,Nv);
% c4_check = zeros(Nu-1,Nv);
% for vi=1:length(v_point_t)
%     RW_u = RW(:,vi);
%     SPu = spline(u_point_t,RW_u);
%     c1_check(:,vi) = SPu.coefs(:,1);
%     c2_check(:,vi) = SPu.coefs(:,2);
%     c3_check(:,vi) = SPu.coefs(:,3);
%     c4_check(:,vi) = SPu.coefs(:,4);
% end
% toc
% Diff_c1 = c1-c1_check;
% Diff_c2 = c2-c2_check;
% Diff_c3 = c3-c3_check;
% Diff_c4 = c4-c4_check;
% 
% t1=find(Diff_c1~=0);
% t2=find(Diff_c2~=0);
% t3=find(Diff_c3~=0);
% t4=find(Diff_c4~=0);
%%%%%---------------this block for checking c1~c4 ---------------

end
