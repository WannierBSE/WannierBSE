%==========================================================================
% STEP3_SHIFT_R_AND_Z
%
% DESCRIPTION:
%   Recenters and shifts Wannier real-space coordinates into the WBSE working convention before normalization.
%
% INPUT ARGUMENTS:
%   None. Input data and paths are obtained from the active preprocessing workspace.
%
% OUTPUT:
%   None. Shifted Wannier arrays and metadata are written to the temporary workspace.
%==========================================================================

[a1,a2,a3,a,tau,N_atoms] = read_structure;

%----------read Wannier functions---------
% Load data from the workspace/WF_raw directory
load(fullfile(Paths.temp_raw, 'W_up_raw.mat'))
load(fullfile(Paths.temp_raw, 'W_down_raw.mat'))
load(fullfile(Paths.temp_raw, 'r.mat'))

GridInfo = importdata(fullfile(Paths.temp_raw, 'GridInfo.txt'));
Atoms = importdata(fullfile(Paths.temp_raw, 'WF_AtomPosition.txt'));

Na1 = GridInfo(1,1);
Na2 = GridInfo(1,2);
Na3 = GridInfo(1,3);
A1s = GridInfo(3,:);
A2s = GridInfo(4,:);
A3s = GridInfo(5,:);
%----------read Wannier functions---------

r_origin = r;

%--------------shift r outside supercell---------
% OPT-INSTRUMENTATION
t_opt_shift_map = tic;
NSC = ceil(norm(A1s)./norm(a1));
A1 = NSC.*a1;
A2 = NSC.*a2;
A3 = NSC.*a3;
tol_shift = 0.05;
Bottom_point = (-((NSC-1)/2)*a1-((NSC-1)/2)*a2)+[0,-tol_shift,0];
Top_point = (((NSC+1)/2)*a1+((NSC+1)/2)*a2)+[0,-tol_shift,0];
Right_point = r_origin(2+Na1,:)+A1+[+0.02,-tol_shift,0];
Left_point = r_origin(2+Na1,:)+A2+[-0.02,-tol_shift,0];
SuperCell_points = [Bottom_point;Right_point;Top_point;Left_point;Bottom_point];
r2 = r;
ind_A1 = 1:Na1;
ind_A2 = 1:Na1:Na1*Na2;
for zi = 1:Na3
    r2(ind_A1+((zi-1)*(Na1*Na2)),1:2) = r2(ind_A1+((zi-1)*(Na1*Na2)),1:2)+A2(1:2);
    r2(ind_A2+((zi-1)*(Na1*Na2)),1:2) = r2(ind_A2+((zi-1)*(Na1*Na2)),1:2)+A1(1:2);
end
in = zeros(size(r,1),1);
for zi=1:Na3
    [in((1:Na1*Na2)+((zi-1)*(Na1*Na2))),~] = inpolygon(r2((1:Na1*Na2)+((zi-1)*(Na1*Na2)),1),r2((1:Na1*Na2)+((zi-1)*(Na1*Na2)),2),SuperCell_points(:,1),SuperCell_points(:,2));
end
out = find(in==0);
%--------------shift r outside supercell---------

%------------change index---------
r2_inplane = r2(1:(Na1*Na2),:);
r_inplane_shifted = zeros(Na1*Na2,3);
for rj=1:Na1
    for ri=1:Na2
        r_inplane_shifted(ri+((rj-1)*Na1),:) = (ri-1).*(A1s./(Na1-1))+(rj-1).*(A2s./(Na2-1));
    end
end
r_inplane_shifted = r_inplane_shifted+r_origin(2+Na1,:);

ind_rearrange = zeros(size(r_inplane_shifted,1),1);
for i=1:length(ind_rearrange)
    dr = sqrt(sum((r_inplane_shifted(i,:)-r2_inplane).^2,2));
    ind_rearrange(i) = find(dr==min(dr));
end
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_shift_map, 'shift_map_build', 'exclusive', num2str(Na1*Na2), sprintf('%dx%d', Na1, Na2));
t_opt_shift_apply = tic;

r_shifted = zeros(size(r2));
r_shifted(:,3) = r2(:,3);
for zi = 1:Na3
    r_shifted((1:(Na1*Na2))+((zi-1)*(Na1*Na2)),1:2) = r2(ind_rearrange+((zi-1)*(Na1*Na2)),1:2);
end

W_up = zeros(size(W_up_raw));
W_down = zeros(size(W_down_raw));
for zi = 1:Na3
    W_up((1:(Na1*Na2))+((zi-1)*(Na1*Na2)),:) = W_up_raw(ind_rearrange+((zi-1)*(Na1*Na2)),:);
    W_down((1:(Na1*Na2))+((zi-1)*(Na1*Na2)),:) = W_down_raw(ind_rearrange+((zi-1)*(Na1*Na2)),:);
end
%------------change index---------

%-------------------shift z points-----------------
tz = find(r_shifted(:,3)<-0.02);
r_shifted(tz,3) = r_shifted(tz,3)+a3(3);

r_shifted_temp = r_shifted(tz,:);
r_shifted(tz,:) = [];
r_shifted = [r_shifted;r_shifted_temp];

W_up_z_temp = W_up(tz,:);
W_up(tz,:) = [];
W_up = [W_up;W_up_z_temp];

W_down_z_temp = W_down(tz,:);
W_down(tz,:) = [];
W_down = [W_down;W_down_z_temp];
%-------------------shift z points-----------------

r = r_shifted;
% OPT-INSTRUMENTATION
opt_wf_timing('stop', t_opt_shift_apply, 'shift_apply', 'exclusive', num2str(size(W_up,2)), sprintf('%dx%d', size(W_up,1), size(W_up,2)));

% Create workspace/WF_raw_shifted if it doesn't exist
if ~exist(Paths.temp_shifted, 'dir')
    mkdir(Paths.temp_shifted);
end

% Save results to the shifted workspace
save(fullfile(Paths.temp_shifted, 'W_up.mat'),'W_up','-v7.3')
save(fullfile(Paths.temp_shifted, 'W_down.mat'),'W_down','-v7.3')
save(fullfile(Paths.temp_shifted, 'r.mat'),'r','-v7.3')

% Copy metadata files
copyfile(fullfile(Paths.temp_raw, 'GridInfo.txt'), fullfile(Paths.temp_shifted, 'GridInfo.txt'))
copyfile(fullfile(Paths.temp_raw, 'WF_AtomPosition.txt'), fullfile(Paths.temp_shifted, 'WF_AtomPosition.txt'))

%-------delete data in previous step (WF_raw)--------
if exist(fullfile(Paths.temp_raw, 'W_up_raw.mat'), 'file'), delete(fullfile(Paths.temp_raw, 'W_up_raw.mat')); end
if exist(fullfile(Paths.temp_raw, 'W_down_raw.mat'), 'file'), delete(fullfile(Paths.temp_raw, 'W_down_raw.mat')); end
if exist(fullfile(Paths.temp_raw, 'r.mat'), 'file'), delete(fullfile(Paths.temp_raw, 'r.mat')); end
%-------delete data in previous step--------
