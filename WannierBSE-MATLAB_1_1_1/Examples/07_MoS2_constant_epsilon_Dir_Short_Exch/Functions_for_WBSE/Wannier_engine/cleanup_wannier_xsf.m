function cleanup_wannier_xsf(root)
%==========================================================================
% CLEANUP_WANNIER_XSF
%
% DESCRIPTION:
%   Removes input Wannier XSF files at successful shutdown. Only
%   wannier90_*.xsf files in the four expected spin-component folders are
%   considered; unrelated files and data outside that subtree are preserved.
%
% INPUT ARGUMENTS:
%   root : optional XSF root directory. The configured project path is used
%       when omitted.
%
% OUTPUT:
%   None. Matching XSF input files are deleted.
%==========================================================================

if nargin < 1 || isempty(root)
    Paths = init_wannier_paths();
    root = Paths.xsf_root;
end

fprintf('[INFO ] Wannier: Removing input XSF files after successful completion...\n');
subfolders = {'up_real', 'up_imag', 'down_real', 'down_imag'};
for iFolder = 1:numel(subfolders)
    target_dir = fullfile(root, subfolders{iFolder});
    if ~isfolder(target_dir)
        continue;
    end
    files = dir(fullfile(target_dir, 'wannier90_*.xsf'));
    for iFile = 1:numel(files)
        delete(fullfile(files(iFile).folder, files(iFile).name));
    end
end
end
