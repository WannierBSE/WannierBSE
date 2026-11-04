function reorder_WF_centers(wf_centers_in, wf_centers_out)
%==========================================================================
% REORDER_WF_CENTERS
%
% DESCRIPTION:
%   Reorders WF center metadata to the WBSE spin-block convention and writes explicit spinor labels.
%
% INPUT ARGUMENTS:
%   wf_centers_in : input WF_centers.txt path.
%   wf_centers_out : output path for reordered metadata.
%
% OUTPUT:
%   None. The reordered center metadata is written to wf_centers_out.
%==========================================================================

    if ~isfile(wf_centers_in)
        error('reorder_WF_centers:FileNotFound', 'WF_centers.txt not found at %s', wf_centers_in);
    end

    wf = read_WF_center(wf_centers_in);
    reorder_map = spin_block_reorder_map(wf.spinor);
    NB = numel(wf.index);

    out_index = (1:NB).';
    out_atom = zeros(NB, 1);
    out_spin = strings(NB, 1);
    for old = 1:NB
        new = reorder_map(old);
        out_atom(new) = wf.atom(old);
        out_spin(new) = wf.spinor(old);
    end

    fid = fopen(wf_centers_out, 'w');
    if fid == -1
        error('reorder_WF_centers:CannotWrite', 'Could not open %s for writing.', wf_centers_out);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, '   j     I     Spinor\n');
    for n = 1:NB
        fprintf(fid, '  %3d   %3d     %s\n', out_index(n), out_atom(n), char(out_spin(n)));
    end

    fprintf('  -> WF centers exported with explicit spinor labels.\n');
end
