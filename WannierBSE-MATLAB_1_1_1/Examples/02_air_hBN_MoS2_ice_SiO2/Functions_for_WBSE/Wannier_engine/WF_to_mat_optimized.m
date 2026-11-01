function WF_to_mat_optimized()
%==========================================================================
% WF_TO_MAT_OPTIMIZED
%
% DESCRIPTION:
%   Builds one transactional Wannier cache using the fused optimized
%   preprocessing implementation without intermediate component MAT files.
%
% INPUT ARGUMENTS:
%   None. Project paths and preprocessing controls are loaded from the
%   standardized project configuration.
%
% OUTPUT:
%   None. A validated transactional Wannier cache is written to the project.
%==========================================================================
Paths=init_wannier_paths(); root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
param_file=fullfile(Paths.parameters,'exchange_control.txt'); p=read_params(param_file);
cleanup_xsf=false; if isfield(p,'cleanup_xsf'), cleanup_xsf=parse_cleanup_xsf(p.cleanup_xsf); end
Validate_WF_Overlap=false; if isfield(p,'Validate_WF_Overlap'), Validate_WF_Overlap=parse_optional_boolean(p.Validate_WF_Overlap,'Validate_WF_Overlap',param_file); end
final=Paths.final_output;
if isfolder(final), error('WFOptimized:OutputExists','Refusing to overwrite existing WF cache: %s',final); end
stamp=datestr(now,'yyyymmdd_HHMMSSFFF'); incomplete=[final '.incomplete_' stamp];
if isfolder(incomplete), error('WFOptimized:IncompleteExists','Refusing existing incomplete cache: %s',incomplete); end
mkdir(incomplete);
fprintf('[PRE  ] Wannier: optimized fused preprocessing...\n'); % OPT-INSTRUMENTATION
total=tic;
tcache=tic; opt_wf_timing('stop',tcache,'cache_decision','exclusive','1','NA');
tg=tic; g=wf_prepare_geometry_optimized(root); setup=toc(tg);
% OPT-INSTRUMENTATION
% OPT-INSTRUMENTATION
% OPT-INSTRUMENTATION
nb=22; map=[1;12;2;13;3;14;4;15;5;16;6;17;7;18;8;19;9;20;10;21;11;22];
W_up=zeros(g.n,nb,'like',1i); W_down=zeros(g.n,nb,'like',1i); written=false(1,nb); dfile=fopen(fullfile(incomplete,'normalization_diagnostics.csv'),'w'); fprintf(dfile,'source_wf,destination_column,norm_before,norm_const,norm_deviation\n');
for source=1:nb
    c=wf_process_one_optimized(g,source); dest=map(source);
    if written(dest), error('WFOptimized:DuplicateDestination','Destination %d repeated.',dest); end
    if ~isequal(size(c.W_up),[g.n 1])||~isequal(size(c.W_down),[g.n 1])||~isa(c.W_up,'double')||~isa(c.W_down,'double')||isreal(c.W_up)||isreal(c.W_down)||any(~isfinite(c.W_up))||any(~isfinite(c.W_down)), error('WFOptimized:InvalidColumn','WF %d failed output checks.',source); end
    dev=abs(c.norm_before-c.norm_const^2); if dev>1e-11, error('WFOptimized:Normalization','WF %d deviation %.17g.',source,dev); end
    W_up(:,dest)=c.W_up; W_down(:,dest)=c.W_down; written(dest)=true;
    fprintf(dfile,'%d,%d,%.17g,%.17g,%.17g\n',source,dest,c.norm_before,c.norm_const,dev);
    % OPT-INSTRUMENTATION
drawnow; clear c
end
fclose(dfile); if ~all(written), error('WFOptimized:Incomplete','Not all output columns were written.'); end
if any(~isfinite(W_up),'all')||any(~isfinite(W_down),'all'), error('WFOptimized:Nonfinite','Final WF matrix contains NaN/Inf.'); end
up_path=fullfile(incomplete,'WF_up.mat'); down_path=fullfile(incomplete,'WF_down.mat'); tu=tic; save(up_path,'W_up','-v7.3','-nocompression'); % OPT-INSTRUMENTATION
td=tic; save(down_path,'W_down','-v7.3','-nocompression'); % OPT-INSTRUMENTATION
r=g.r; save(fullfile(incomplete,'r.mat'),'r','-v7.3'); wf_write_gridinfo(fullfile(incomplete,'GridInfo.txt'),g); wf_write_atoms(fullfile(incomplete,'WF_AtomPosition.txt'),g); centers_candidate=fullfile(incomplete,'WF_centers_WBSE.txt'); reorder_WF_centers(fullfile(Paths.parameters,'WF_centers.txt'),centers_candidate);
if Validate_WF_Overlap
    % OPT-INSTRUMENTATION
    wf_overlap_diagnostic(W_up,W_down,g,fullfile(incomplete,'M.mat'));
else
    % OPT-INSTRUMENTATION
end
[ok,why]=validate_wf_cache(incomplete); if ~ok,error('WFOptimized:Validation','Incomplete cache: %s',why);end
wf_write_text(fullfile(incomplete,'COMPLETE'),'Optimized WF cache complete.\n'); [ok,msg]=movefile(incomplete,final); if ~ok,error('WFOptimized:Finalize','%s',msg);end
% OPT-INSTRUMENTATION
fprintf('[DONE ] Wannier: optimized preprocessing completed in %.2f s. Cleanup flag: %d\n',toc(total),cleanup_xsf);
end
%--------------------------------------------------------------------------
% INTERNAL HELPER: WF_WRITE_TEXT
%
% PURPOSE:
%   Internal helper used by WF_to_mat_optimized and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function wf_write_text(p,s),f=fopen(p,'w');if f<0,error('WFOptimized:Write','Cannot write %s',p);end;fprintf(f,'%s',s);fclose(f);end
%--------------------------------------------------------------------------
% INTERNAL HELPER: WF_WRITE_GRIDINFO
%
% PURPOSE:
%   Internal helper used by WF_to_mat_optimized and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function wf_write_gridinfo(p,g),f=fopen(p,'w');if f<0,error('WFOptimized:Metadata','Cannot write GridInfo');end;fprintf(f,'%d %d %d \n',g.dims);fprintf(f,'%.14f %.14f %.14f \n',g.origin);for k=1:3,fprintf(f,'%.14f %.14f %.14f',g.vectors(k,:));if k<3,fprintf(f,' \n');end,end;fclose(f);end
%--------------------------------------------------------------------------
% INTERNAL HELPER: WF_WRITE_ATOMS
%
% PURPOSE:
%   Internal helper used by WF_to_mat_optimized and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function wf_write_atoms(p,g),f=fopen(p,'w');if f<0,error('WFOptimized:Metadata','Cannot write atoms');end;fprintf(f,'%.14f %.14f %.14f \n',g.origin);for k=1:numel(g.atoms),fprintf(f,'%.14f %.14f %.14f \n',g.atoms{k}.position);end,fclose(f);end
%--------------------------------------------------------------------------
% INTERNAL HELPER: WF_OVERLAP_DIAGNOSTIC
%
% PURPOSE:
%   Internal helper used by WF_to_mat_optimized and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function wf_overlap_diagnostic(U,D,g,p),n=size(U,2);M=zeros(n,n);for i=1:n,for j=1:n,M(i,j)=wf_overlap_integral(U(:,i),D(:,i),U(:,j),D(:,j),g);end,end;save(p,'M','-v7.3');off=M;off(1:n+1:end)=0;fprintf('  -> WF overlap diagnostic: max diagonal deviation %.6e, max off-diagonal %.6e, Hermiticity residual %.6e, nonfinite entries %d\n',max(abs(diag(M)-1)),max(abs(off(:))),norm(M-M','fro')/max(norm(M,'fro'),eps),sum(~isfinite(M(:))));end
%--------------------------------------------------------------------------
% INTERNAL HELPER: WF_OVERLAP_INTEGRAL
%
% PURPOSE:
%   Internal helper used by WF_to_mat_optimized and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function S=wf_overlap_integral(au,ad,bu,bd,g),W=conj(au).*bu+conj(ad).*bd;R=permute(reshape(W,g.dims),[2 3 1]);P=spline(g.u_point,R);C=cell(4,1);for k=1:4,C{k}=permute(reshape(P.coefs(:,k),[g.dims(2),g.dims(3),g.dims(1)-1]),[3 1 2]);end;Su=g.J*sum(g.hf0(C{1},C{2},C{3},C{4},g.u_point(1:end-1),g.hu),1);P=spline(g.v_point,transpose(squeeze(Su)));C=cell(4,1);for k=1:4,C{k}=transpose(reshape(P.coefs(:,k),[g.dims(3),g.dims(2)-1]));end;Sv=sum(g.hf0(C{1},C{2},C{3},C{4},g.v_point(1:end-1),g.hv),1);P=spline(g.z_point,Sv);S=sum(g.hf0(P.coefs(:,1),P.coefs(:,2),P.coefs(:,3),P.coefs(:,4),g.z_point(1:end-1),g.hz));end
