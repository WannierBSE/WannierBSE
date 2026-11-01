function reorder_map = spin_block_reorder_map(spinor)
%==========================================================================
% SPIN_BLOCK_REORDER_MAP
%
% DESCRIPTION:
%   Builds a permutation that groups up-spin Wannier functions before down-spin Wannier functions.
%
% INPUT ARGUMENTS:
%   spinor : spinor label vector.
%
% OUTPUT:
%   reorder_map : old-index to new-index permutation.
%==========================================================================

    up = find(spinor == "up");
    down = find(spinor == "down");
    order_old = [up(:); down(:)];
    reorder_map = zeros(numel(spinor), 1);
    for new = 1:numel(order_old)
        reorder_map(order_old(new)) = new;
    end
end
