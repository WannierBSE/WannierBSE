function C_reordered = reorder_TB_basis(C_original, reorder_map)
%==========================================================================
% REORDER_TB_BASIS
%
% DESCRIPTION:
%   Reorders TB eigenvector coefficients from Wannier90 ordering to the WBSE spin-block convention.
%
% INPUT ARGUMENTS:
%   C_original : coefficient array with dimensions N_kpt-by-NB-by-N_bands.
%   reorder_map : optional permutation from original basis index to WBSE index.
%
% OUTPUT:
%   C_reordered : coefficient array in WBSE basis order.
%==========================================================================

    % Determine the basis dimension from the coefficient tensor and fall
    % back to the legacy reorder only when the caller has no explicit map.
    [~, NB, ~] = size(C_original);

    if nargin < 2 || isempty(reorder_map)
        reorder_map = default_reorder_map(NB);
    end

    reorder_map = reorder_map(:);
    if numel(reorder_map) ~= NB || any(sort(reorder_map) ~= (1:NB).')
        error('reorder_TB_basis:InvalidReorderMap', ...
              'Basis reorder map must be a permutation of 1:NB.');
    end

    % Copy each original coefficient column into its WBSE spin-block
    % position while preserving k-point and band dimensions.
    C_reordered = zeros(size(C_original));
    for n = 1:NB
        C_reordered(:, reorder_map(n), :) = C_original(:, n, :);
    end
end

function reorder_map = default_reorder_map(NB)
%==========================================================================
% DEFAULT_REORDER_MAP
%
% DESCRIPTION:
%   Builds the legacy 22-band odd/even spinor reorder map when no explicit map is supplied.
%
% OUTPUT:
%   reorder_map : basis permutation for WBSE spin blocks.
%==========================================================================

    % The 22-band model historically interleaves spinor partners; all
    % other dimensions are assumed to already be in the desired order.
    if NB == 22
        reorder_map = [1; 12; 2; 13; 3; 14; 4; 15; 5; 16; 6; 17; ...
                       7; 18; 8; 19; 9; 20; 10; 21; 11; 22];
    else
        reorder_map = (1:NB).';
    end
end
