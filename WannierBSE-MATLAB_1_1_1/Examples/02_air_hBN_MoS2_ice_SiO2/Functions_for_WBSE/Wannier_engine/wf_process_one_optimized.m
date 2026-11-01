function r=wf_process_one_optimized(g,wf)
%==========================================================================
% WF_PROCESS_ONE_OPTIMIZED
%
% DESCRIPTION:
%   Processes one Wannier function through the candidate fused optimized
%   path without writing the production cache.
%
% INPUT ARGUMENTS:
%   g : shared optimized Wannier geometry structure.
%   wf : one-based Wannier-function index.
%
% OUTPUT:
%   r : structure containing the processed spinor columns and diagnostics.
%==========================================================================
if wf<1||wf~=floor(wf),error('Stage5:WF','Invalid WF index.');end;t0=tic;nm={'up_real','up_imag','down_real','down_imag'};p=cell(4,1);tr=tic;
for k=1:4,p{k}=read_xsf_wf_fast(fullfile(g.xsf_root,nm{k},sprintf('wannier90_%05d.xsf',wf)),'textscan');if ~isequal(p{k}.dims,g.dims)||~isequal(p{k}.origin,g.origin)||~isequal(p{k}.vectors,g.vectors)||~wf_atoms_equal(p{k}.atoms,g.atoms),error('Stage5:WFGeometry','WF component geometry differs.');end,end
read_s=toc(tr);ta=tic;up=p{1}.values+1i*p{2}.values;dn=p{3}.values+1i*p{4}.values;assembly_s=toc(ta);clear p
ts=tic;up=up(g.source_order);dn=dn(g.source_order);shift_s=toc(ts);ti=tic;[before,nc]=wf_integral(up,dn,g);integral_s=toc(ti);tc=tic;up=up/nc;dn=dn/nc;scale_s=toc(tc);dest=[1;12;2;13;3;14;4;15;5;16;6;17;7;18;8;19;9;20;10;21;11;22];dest=dest(wf);trr=tic;reorder_s=toc(trr);r=struct('wf_index',wf,'destination',dest,'W_up',up,'W_down',dn,'norm_before',before,'norm_const',nc,'norm_after_direct',sum(abs(up).^2+abs(dn).^2),'timing',struct('four_file_read_s',read_s,'spinor_assembly_s',assembly_s,'shift_apply_s',shift_s,'normalization_integral_s',integral_s,'normalization_scale_s',scale_s,'basis_reorder_s',reorder_s,'remaining_s',toc(t0)-read_s-assembly_s-shift_s-integral_s-scale_s-reorder_s,'candidate_total_s',toc(t0)));
end
function [S,nc]=wf_integral(up,dn,g)
W=(conj(up).*up)+(conj(dn).*dn);R=permute(reshape(W,g.dims),[2 3 1]);P=spline(g.u_point,R);c=cell(4,1);for k=1:4,c{k}=permute(reshape(P.coefs(:,k),[g.dims(2),g.dims(3),g.dims(1)-1]),[3 1 2]);end;Su=g.J*sum(g.hf0(c{1},c{2},c{3},c{4},g.u_point(1:end-1),g.hu),1);P=spline(g.v_point,transpose(squeeze(Su)));c=cell(4,1);for k=1:4,c{k}=transpose(reshape(P.coefs(:,k),[g.dims(3),g.dims(2)-1]));end;Sv=sum(g.hf0(c{1},c{2},c{3},c{4},g.v_point(1:end-1),g.hv),1);P=spline(g.z_point,Sv);S=sum(g.hf0(P.coefs(:,1),P.coefs(:,2),P.coefs(:,3),P.coefs(:,4),g.z_point(1:end-1),g.hz));nc=sqrt(S);
end

function e=wf_atoms_equal(a,b),e=numel(a)==numel(b);for k=1:min(numel(a),numel(b)),e=e&&strcmp(a{k}.label,b{k}.label)&&isequal(a{k}.position,b{k}.position);end,end

