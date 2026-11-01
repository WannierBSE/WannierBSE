function ind=wf_direct_shift_map(dims,origin,A1s,A2s,a1,a2)
%==========================================================================
% WF_DIRECT_SHIFT_MAP
%
% DESCRIPTION:
%   Candidate-only direct equivalent of the legacy in-plane nearest-point
%   map used by optimized Wannier preprocessing.
%
% INPUT ARGUMENTS:
%   dims, origin : structured-grid dimensions and origin.
%   A1s, A2s : supercell in-plane vectors.
%   a1, a2 : unit-cell in-plane lattice vectors.
%
% OUTPUT:
%   ind : source-index map from target in-plane grid points.
%==========================================================================
nx=dims(1);ny=dims(2);if nx<2||ny<2||any(dims~=floor(dims))||any(~isfinite([origin A1s A2s a1 a2])),error('Stage7C:Grid','Invalid structured grid.');end
nsc=ceil(norm(A1s)/norm(a1));B1=nsc*a1;B2=nsc*a2;b1=A1s/(nx-1);b2=A2s/(ny-1);basis=[b1(1:2);b2(1:2)];if abs(det(basis))<eps,error('Stage7C:Grid','Degenerate in-plane lattice.');end
anchor=origin+b1+b2;ind=zeros(nx*ny,1);tol=max(1e-10,1e-9*max([norm(A1s) norm(A2s) 1]));
for j=0:ny-1
 for i=0:nx-1
  target=anchor+i*b1+j*b2;hits=zeros(4,1);residuals=zeros(4,1);nh=0;
  for sx=0:1
   for sy=0:1
    q=(target-origin-sx*B1-sy*B2);ij=q(1:2)/basis;ix=round(ij(1));iy=round(ij(2));valid=ix>=0&&ix<nx&&iy>=0&&iy<ny&&((ix==0)==logical(sx))&&((iy==0)==logical(sy));
    if valid
     residual=norm(origin+ix*b1+iy*b2+sx*B1+sy*B2-target);nh=nh+1;hits(nh)=ix+iy*nx+1;residuals(nh)=residual;
    end
   end
  end
  if nh==0,error('Stage7C:Map','No direct source for target (%d,%d).',i,j);end;best=min(residuals(1:nh));ind(i+1+j*nx)=min(hits(residuals(1:nh)<=best+tol));
 end
end
end

