function externalTB = validate_external_kmesh_tb_input(UserInputPath, Nv, Nc, Exchange_Interaction)
%==========================================================================
% VALIDATE_EXTERNAL_KMESH_TB_INPUT
%
% DESCRIPTION:
%   Validates whether user-provided k-mesh and TB band files can replace internally generated WBSE inputs.
%
% INPUT ARGUMENTS:
%   UserInputPath : user input directory.
%   Nv, Nc : required valence and conduction band counts.
%   Exchange_Interaction : exchange flag used to reject unsupported external meshes.
%
% OUTPUT:
%   externalTB : structure describing validation status and resolved paths.
%==========================================================================

    userKmeshPath = fullfile(UserInputPath, 'kmesh.txt');
    userBandsPath = fullfile(UserInputPath, 'TB_data');
    hasUserKmesh = isfile(userKmeshPath);
    hasUserTBFolder = isfolder(userBandsPath);
    [hasCompleteUserTB, missingUserBandFiles] = check_user_tb_data(userBandsPath, Nv, Nc);
    useUserProvidedKmeshTB = hasUserKmesh && hasCompleteUserTB;

    if ~hasUserKmesh && hasUserTBFolder
        error('WBSE:ExternalInputPairRequired', ...
              '[ERROR] User TB_data requires a matching user kmesh. Provide User_input/kmesh.txt or remove User_input/TB_data.');
    end

    if useUserProvidedKmeshTB && Exchange_Interaction
        error('WBSE:ExternalTBExchangeUnsupported', ...
              ['User-provided kmesh/TB_data is supported only for direct-interaction calculations. ', ...
               'Set Exchange_Interaction = false, or use Wannier90-based inputs so WBSE can generate ', ...
               'consistent Wannier functions and exchange tensors.']);
    end

    externalTB = struct();
    externalTB.useUserProvidedKmeshTB = useUserProvidedKmeshTB;
    externalTB.userKmeshPath = userKmeshPath;
    externalTB.userBandsPath = userBandsPath;
    externalTB.hasUserKmesh = hasUserKmesh;
    externalTB.hasUserTBFolder = hasUserTBFolder;
    externalTB.hasCompleteUserTB = hasCompleteUserTB;
    externalTB.missingUserBandFiles = missingUserBandFiles;
end

function [files_exist, missing_files] = check_user_tb_data(userBandsPath, Nv, Nc)
%==========================================================================
% CHECK_USER_TB_DATA
%
% DESCRIPTION:
%   Checks that every required user TB band file exists.
%
% OUTPUT:
%   files_exist : true when all files are present.
%   missing_files : cell array of missing file names.
%==========================================================================

    required_files = [arrayfun(@(n) sprintf('v%d_TB.txt', n), 1:Nv, 'UniformOutput', false), ...
                      arrayfun(@(n) sprintf('c%d_TB.txt', n), 1:Nc, 'UniformOutput', false)];
    missing_files = {};
    if ~isfolder(userBandsPath)
        missing_files = required_files;
        files_exist = false;
        return;
    end
    for n = 1:numel(required_files)
        if ~isfile(fullfile(userBandsPath, required_files{n}))
            missing_files{end + 1} = required_files{n}; %#ok<AGROW>
        end
    end
    files_exist = isempty(missing_files);
end

function text = format_missing_files(missing_files)
%==========================================================================
% FORMAT_MISSING_FILES
%
% DESCRIPTION:
%   Formats a missing-file list for validation error messages.
%
% OUTPUT:
%   text : comma-separated file list.
%==========================================================================

    if isempty(missing_files)
        text = '<none>';
    else
        text = strjoin(missing_files, ', ');
    end
end
