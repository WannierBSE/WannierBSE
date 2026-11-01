function g=wf_prepare_geometry_optimized(root)
%==========================================================================
% WF_PREPARE_GEOMETRY_OPTIMIZED
%
% DESCRIPTION:
%   Prepares shared geometry and interpolation data for the optimized fused
%   Wannier preprocessing workflow. No production cache is written here.
%
% INPUT ARGUMENTS:
%   root : project root directory.
%
% OUTPUT:
%   g : structure containing geometry, reordered grid data, and timings.
%==========================================================================
ui=fullfile(root,'User_input'); xr=fullfile(ui,'Wannier_functions_xsf');
cn={'up_real','up_imag','down_real','down_imag'}; h=cell(4,1);
for k=1:4,h{k}=read_xsf_wf_fast(fullfile(xr,cn{k},'wannier90_00001.xsf'),'textscan');end
for k=2:4
 if ~isequal(h{1}.dims,h{k}.dims)||~isequal(h{1}.origin,h{k}.origin)||~isequal(h{1}.vectors,h{k}.vectors)||~wf_atoms_equal(h{1}.atoms,h{k}.atoms),error('WFOptimized:GeometryMismatch','Component XSF geometry differs.');end
end
d=h{1}.dims; nx=d(1);ny=d(2);nz=d(3); n=prod(d); nxy=nx*ny; o=h{1}.origin; A1s=h{1}.vectors(1,:);A2s=h{1}.vectors(2,:);A3s=h{1}.vectors(3,:);
x=(1:nx)';xr1=repmat(x,ny*nz,1);y=(1:ny)';yr=repmat(repelem(y,nx),nz,1);z=(1:nz)';zr=repelem(z,nxy);
r0=zeros(n,3);r0(:,1)=A1s(1)*(xr1-1)/(nx-1)+A2s(1)*(yr-1)/(ny-1)+A3s(1)*(zr-1)/(nz-1)+o(1);r0(:,2)=A1s(2)*(xr1-1)/(nx-1)+A2s(2)*(yr-1)/(ny-1)+A3s(2)*(zr-1)/(nz-1)+o(2);r0(:,3)=A1s(3)*(xr1-1)/(nx-1)+A2s(3)*(yr-1)/(ny-1)+A3s(3)*(zr-1)/(nz-1)+o(3);
[a1,a2,a3]=read_structure(ui); r2=r0; NSC=ceil(norm(A1s)/norm(a1)); B1=NSC*a1; B2=NSC*a2;
ia1=1:nx;ia2=1:nx:nxy;
for zi=1:nz
 r2(ia1+(zi-1)*nxy,1:2)=r2(ia1+(zi-1)*nxy,1:2)+B2(1:2);r2(ia2+(zi-1)*nxy,1:2)=r2(ia2+(zi-1)*nxy,1:2)+B1(1:2);
end
want=zeros(nxy,3);for rj=1:nx,for ri=1:ny,want(ri+(rj-1)*nx,:)=(ri-1)*(A1s/(nx-1))+(rj-1)*(A2s/(ny-1));end,end;want=want+r0(2+nx,:);
tmap=tic;ip=wf_direct_shift_map(d,o,A1s,A2s,a1,a2);shift_map_s=toc(tmap); if max(abs(want(:,1:2)-r2(ip,1:2)),[],'all')>1e-7,error('WFOptimized:MapResidual','Shift map residual too large.');end
bo=zeros(n,1);rpre=zeros(n,3);rpre(:,3)=r2(:,3);for zi=1:nz,b=(1:nxy)+(zi-1)*nxy;bo(b)=ip+(zi-1)*nxy;rpre(b,1:2)=r2(ip+(zi-1)*nxy,1:2);end;tz=find(rpre(:,3)<-.02);rpre(tz,3)=rpre(tz,3)+a3(3);keep=true(n,1);keep(tz)=false;so=[bo(keep);bo(tz)];r=rpre([find(keep);tz],:);
tn=tic;[J,hu,hv,hz,u,v,up,vp,zp]=wf_coord(r,a1,a2,nz);[~,hf0]=SPint_uv;normalization_geometry_s=toc(tn);
g=struct();g.repo_root=root;g.xsf_root=xr;g.components=cn;g.dims=d;g.n=n;g.origin=o;g.vectors=h{1}.vectors;g.atoms=h{1}.atoms;g.a1=a1;g.a2=a2;g.a3=a3;g.r=r;g.source_order=so;g.ind_inplane=ip;g.tz_count=numel(tz);g.J=J;g.hu=hu;g.hv=hv;g.hz=hz;g.u=u;g.v=v;g.u_point=up;g.v_point=vp;g.z_point=zp;g.hf0=hf0;
g.timing=struct('shift_map_s',shift_map_s,'normalization_geometry_s',normalization_geometry_s);g.explicit_bytes=8*(numel(r0)+numel(r2)+numel(want)+numel(so)+numel(r)+numel(u)+numel(v)+numel(up)+numel(vp)+numel(zp));
end
function [J,hu,hv,hz,u,v,up,vp,zp]=wf_coord(r,a1,a2,Nz)
a1x=a1(1);a1y=a1(2);a2x=a2(1);a2y=a2(2);x=r(:,1);y=r(:,2);z=r(:,3);u=x-(a2x/a2y).*y;v=-x+(a1x/a1y).*y;up=uniquetol(u,1e-6);vp=uniquetol(v,1e-6);zp=transpose(linspace(min(z),max(z),Nz));hu=up(2)-up(1);hv=vp(2)-vp(1);hz=(max(z)-min(z))/(Nz-1);be=-a2x*a1y+a1x*a2y;J=(a1x*a1y*a2y^2-a2x*a1y^2*a2y)/be^2;
end
function e=wf_atoms_equal(a,b),e=numel(a)==numel(b);for k=1:min(numel(a),numel(b)),e=e&&strcmp(a{k}.label,b{k}.label)&&isequal(a{k}.position,b{k}.position);end,end

