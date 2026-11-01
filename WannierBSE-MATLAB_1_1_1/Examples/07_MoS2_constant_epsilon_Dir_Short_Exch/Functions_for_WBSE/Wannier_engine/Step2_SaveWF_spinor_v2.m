%==========================================================================
% STEP2_SAVEWF_SPINOR_V2
%
% DESCRIPTION:
%   Converts raw spinor Wannier XSF files into MATLAB component arrays and grid metadata inside the temporary Wannier workspace.
%
% INPUT ARGUMENTS:
%   None. Input and output locations are provided through the Paths structure.
%
% OUTPUT:
%   None. Component arrays and grid metadata are written to the temporary workspace.
%==========================================================================

fid_xsf = fopen(fullfile(Paths.xsf_root, 'up_real', 'wannier90_00001.xsf'), 'r');
for i=1:100
    xsf_text{i} = fgetl(fid_xsf);
end
fclose(fid_xsf);

Start_line = 15;
N_atoms = str2num(xsf_text{Start_line});
N_atoms = N_atoms(1);

Atom_Position = zeros(N_atoms,3);
for i=1:N_atoms
    Atom_Position(i,:) = str2num(xsf_text{Start_line+i}(5:end));
end

Number_of_grid = str2num(xsf_text{Start_line+N_atoms+6});
Na1 = Number_of_grid(1);
Na2 = Number_of_grid(2);
Na3 = Number_of_grid(3);
Origin_point = str2num(xsf_text{Start_line+N_atoms+7});
a1 = str2num(xsf_text{Start_line+N_atoms+8});
a2 = str2num(xsf_text{Start_line+N_atoms+9});
a3 = str2num(xsf_text{Start_line+N_atoms+10});
%----------read xsf file--------

% Path updated to workspace/Output
target_dir_up_real = fullfile(Paths.temp_output, 'up_real');
NB = length(dir(fullfile(target_dir_up_real, '*.mat')));

W_up_raw = zeros(Na1*Na2*Na3,NB);
W_down_raw = zeros(Na1*Na2*Na3,NB);

sub_spin = {'up_real', 'down_real', 'up_imag', 'down_imag'};

for spin_index = 1:2
    t_spin = tic;
    for i=1:NB 
        % Load real part
        if spin_index==1
            current_dir_re = fullfile(Paths.temp_output, 'up_real');
        elseif spin_index==2
            current_dir_re = fullfile(Paths.temp_output, 'down_real');
        end

        WF_dataname = fullfile(current_dir_re, [num2str(i),'_WF.mat']);
        WF_real = importdata(WF_dataname);
        
        % Load imag part
        if spin_index==1
            current_dir_im = fullfile(Paths.temp_output, 'up_imag');
        elseif spin_index==2
            current_dir_im = fullfile(Paths.temp_output, 'down_imag');
        end
        
        WF_dataname = fullfile(current_dir_im, [num2str(i),'_WF.mat']);
        WF_imag = importdata(WF_dataname);
        
        if spin_index==1
            W_up_raw(:,i) = WF_real(:,7)+1i*WF_imag(:,7);
        elseif spin_index==2
            W_down_raw(:,i) = WF_real(:,7)+1i*WF_imag(:,7);
        end
        
        clear WF_real WF_imag
    end
    fprintf('  -> Spinor component %d/2 assembled: %d orbitals (%.2f s)\n', spin_index, NB, toc(t_spin));
end

% Load coordinates for r.mat
WF_dataname_r = fullfile(Paths.temp_output, 'up_real', [num2str(NB),'_WF.mat']);
WF_temp = importdata(WF_dataname_r);
r = WF_temp(:,4:6);

% Save results in workspace/WF_raw
if ~exist(Paths.temp_raw, 'dir')
    mkdir(Paths.temp_raw);
end

save(fullfile(Paths.temp_raw, 'W_up_raw.mat'),'W_up_raw','-v7.3')
save(fullfile(Paths.temp_raw, 'W_down_raw.mat'),'W_down_raw','-v7.3')
save(fullfile(Paths.temp_raw, 'r.mat'),'r','-v7.3')

%-------------GridInfo---------------
fid = fopen(fullfile(Paths.temp_raw, 'GridInfo.txt'),'w');
fprintf(fid,'%i %i %i \n',Number_of_grid);
fprintf(fid,'%16.14f %16.14f %16.14f \n',Origin_point);
fprintf(fid,'%16.14f %16.14f %16.14f \n',a1);
fprintf(fid,'%16.14f %16.14f %16.14f \n',a2);
fprintf(fid,'%16.14f %16.14f %16.14f',a3);
fclose(fid);
%-------------GridInfo---------------

%-------------Atom position---------------
fid = fopen(fullfile(Paths.temp_raw, 'WF_AtomPosition.txt'),'w');
fprintf(fid,'%16.14f %16.14f %16.14f \n',Origin_point);
for i=1:N_atoms
    fprintf(fid,'%16.14f %16.14f %16.14f \n',Atom_Position(i,:));
end
fclose(fid);
%-------------Atom position---------------

% Cleanup intermediate .mat files in Output/
for s_idx = 1:4
    spin_folder = sub_spin{s_idx};
    for i=1:NB 
        WF_file_del = fullfile(Paths.temp_output, spin_folder, [num2str(i),'_WF.mat']);
        if exist(WF_file_del, 'file')
            delete(WF_file_del)
        end
    end
end
