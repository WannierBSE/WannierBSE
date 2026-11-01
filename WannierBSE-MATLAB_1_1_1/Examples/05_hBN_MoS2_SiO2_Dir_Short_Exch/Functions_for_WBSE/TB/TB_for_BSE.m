function TB_for_BSE(context)
%==========================================================================
% TB_FOR_BSE
%
% DESCRIPTION:
%   Generates the tight-binding band files used by the WBSE Hamiltonian.
%   The workflow follows a fixed hierarchy to make the exported bands
%   reproducible when the requested BSE basis size is changed:
%
%     1. Diagonalize the complete Wannier90 tight-binding Hamiltonian at
%        every k-point of the active WBSE mesh, retaining all NB eigenvalues
%        and eigenvectors.
%     2. Reorder the eigenvector coefficients into the WBSE spin-block basis
%        convention defined by the Wannier-function centers.
%     3. Rotate exactly degenerate eigenspaces into deterministic spin-
%        adapted states, preserving the original eigenspaces.
%     4. Build the BSE output window from fixed band indices: valence bands
%        are exported downward from VBM, and conduction bands are exported
%        upward from CBM. The parameters Nv and Nc control only how many of
%        these already available bands are written.
%     5. Apply output-level time-reversal energy consistency and phase
%        alignment before serializing the selected bands.
%
% INPUT ARGUMENTS:
%   context : structure defining kmeshFile, wannierWinFile, wannierHrFile, wfCentersFile, controlFile,
%   wtbControlFile, and tbOutputDir.
%
% OUTPUT:
%   v*_TB_WBSE.txt : valence-band TB files written to context.tbOutputDir.
%   c*_TB_WBSE.txt : conduction-band TB files written to context.tbOutputDir.
%==========================================================================

    %------------------------------------------------------
    % 1. Context and control-parameter ingestion
    %------------------------------------------------------
    context = validate_context(context);
    ensure_dir(context.tbOutputDir);
    t_tb_total = tic;

    main_params = read_params(context.controlFile);
    tb_params = read_params(context.wtbControlFile);
    opts = struct();
    opts.outputNv = main_params.Nv;
    opts.outputNc = main_params.Nc;
    opts.VBM = tb_params.VBM;
    opts.CBM = tb_params.CBM;
    if isfield(main_params, 'Npar_TB')
        opts.Npar_TB = main_params.Npar_TB;
    else
        opts.Npar_TB = 1;
    end
    opts.spin_threshold = read_optional_scalar(context.wtbControlFile, ...
        'TB_spin_threshold', 0.3);
    opts.overlap_degeneracy_gap_eV = read_optional_scalar(context.wtbControlFile, ...
        'TB_overlap_degeneracy_gap_eV', 1.0e-5);
    opts.exact_rotation_gap_eV = read_optional_scalar(context.wtbControlFile, ...
        'TB_exact_rotation_gap_eV', 1.0e-10);
    opts.tr_k_tolerance = read_optional_scalar(context.wtbControlFile, ...
        'TB_time_reversal_k_tolerance', 1.0e-8);
    opts.tr_energy_tolerance_eV = read_optional_scalar(context.wtbControlFile, ...
        'TB_time_reversal_energy_tolerance_eV', 1.0e-6);
    opts.tr_spin_tolerance = read_optional_scalar(context.wtbControlFile, ...
        'TB_time_reversal_spin_tolerance', 0.10);
    opts.d3h_k_tolerance = read_optional_scalar(context.wtbControlFile, ...
        'TB_d3h_k_tolerance', 1.0e-8);
    opts.d3h_energy_tolerance_eV = read_optional_scalar(context.wtbControlFile, ...
        'TB_d3h_energy_tolerance_eV', 1.0e-6);
    opts.d3h_candidate_padding = round(read_optional_scalar(context.wtbControlFile, ...
        'TB_d3h_candidate_padding', 4));
    opts.d3h_max_energy_shift_eV = read_optional_scalar(context.wtbControlFile, ...
        'TB_d3h_max_energy_shift_eV', 0.02);

    %------------------------------------------------------
    % 2. Load k mesh, lattice, and Wannier90 Hamiltonian data
    %------------------------------------------------------
    fprintf('[LOAD ] TB: Reading WBSE k-mesh from %s\n', tb_display_path(context.kmeshFile));
    k = read_wbse_kmesh(context.kmeshFile);

    [PLV, latticeSource] = resolve_lattice_vectors(context);
    fprintf('[LOAD ] TB: Reading %s and %s\n', ...
            tb_display_path(latticeSource), tb_display_path(context.wannierHrFile));
    opts.reciprocal = 2 * pi * inv(PLV).';

    [NB, BLVD, DEGN, HR] = read_wannier90_hr(context.wannierHrFile);
    BLVC = BLVD * PLV;

    % Internal valence/conduction families cover the complete spectrum
    % available below and above the reference band-edge indices.
    opts.selectNv = opts.VBM;
    opts.selectNc = NB - opts.CBM + 1;
    opts.Nv = opts.selectNv;
    opts.Nc = opts.selectNc;

    %------------------------------------------------------
    % 3. Validate basis metadata and prepare spin-block bookkeeping
    %------------------------------------------------------
    wf = read_WF_center(context.wfCentersFile);
    reorder_map = spin_block_reorder_map(wf.spinor);
    validate_inputs(k, NB, opts, reorder_map);

    %------------------------------------------------------
    % 4. Full-spectrum diagonalization and basis conversion
    %------------------------------------------------------
    tbPool = gcp('nocreate');
    if isempty(tbPool)
        tbWorkers = 1;
    else
        tbWorkers = tbPool.NumWorkers;
    end

    t_tb_diag = tic;
    [E_data, C_data] = diagonalize_all_kpoints(k, BLVC, DEGN, HR, NB, opts.Npar_TB);
    time_tb_diag = toc(t_tb_diag);

    t_tb_reorder = tic;
    C_perm = permute(C_data, [3, 1, 2]);
    C_reordered = reorder_TB_basis(C_perm, reorder_map);
    C_data = permute(C_reordered, [2, 3, 1]);
    time_tb_reorder = toc(t_tb_reorder);

    %------------------------------------------------------
    % 5. Degenerate-subspace conditioning and fixed output selection
    %------------------------------------------------------
    t_tb_post = tic;
    spin_blocks = spin_blocks_after_reorder(wf.spinor, reorder_map);
    % The full Wannier Hamiltonian is diagonalized above. The BSE output
    % window is then cut from fixed VBM/CBM band indices, so changing Nv or
    % Nc changes only how many already available bands are printed.
    C_data = spin_adapt_exact_degenerate_clusters(E_data, C_data, spin_blocks, opts);
    outputSelection = select_output_window_by_band_index(E_data, C_data, opts, spin_blocks);
    outputSelection = enforce_conduction_spin_pairing(outputSelection, E_data, opts);
    % Apply output-window consistency checks after the fixed band-index
    % window is chosen, avoiding Nv/Nc-dependent changes to deeper bands.
    outputSelection = enforce_d3h_energy_orbit_consistency(outputSelection, E_data, k, opts);
    outputSelection = refresh_selected_spins(C_data, outputSelection, spin_blocks);
    trPartner = find_time_reversal_partners(k, opts);
    outputSelection = enforce_time_reversal_partner_states( ...
        outputSelection, E_data, C_data, k, opts, trPartner);
    outputSelection = relabel_time_reversal_pairs(outputSelection, E_data, k, opts, trPartner);
    C_data = align_time_reversal_gauge(C_data, E_data, outputSelection, opts);
    outputSelection = refresh_selected_spins(C_data, outputSelection, spin_blocks);
    timeReversal = time_reversal_diagnostics(E_data, outputSelection, opts);
    if ~timeReversal.passed
        error('WannierBSE:TBTimeReversalValidation', ...
            'Time-reversal energy validation failed during TB band generation: %s.', ...
            timeReversal.failure_reason);
    end
    time_tb_post = toc(t_tb_post);

    %------------------------------------------------------
    % 6. Serialize the selected WBSE band files
    %------------------------------------------------------
    t_tb_write = tic;
    write_selected_band_files(context.tbOutputDir, k, E_data, C_data, outputSelection, opts);
    time_tb_write = toc(t_tb_write);
    fprintf('[DONE ] TB: Wrote WBSE band files to %s in %.2f s\n', ...
            tb_display_path(context.tbOutputDir), toc(t_tb_total));
end

function context = validate_context(context)
    % Verify the caller supplied every path required by the TB pipeline.
    required = {'kmeshFile', 'wannierWinFile', 'wannierHrFile', 'wfCentersFile', ...
                'controlFile', 'wtbControlFile', 'tbOutputDir'};
    for i = 1:numel(required)
        if ~isfield(context, required{i}) || isempty(context.(required{i}))
            error('WannierBSE:TBMissingContext', ...
                  'Missing required context field "%s".', required{i});
        end
    end
    if ~isfield(context, 'structureFile') || isempty(context.structureFile)
        context.structureFile = fullfile(fileparts(context.wannierWinFile), 'structure.txt');
    end
    if ~isfield(context, 'exchangeInteraction') || isempty(context.exchangeInteraction)
        context.exchangeInteraction = true;
    end

    input_files = {'kmeshFile', 'wannierHrFile', 'wfCentersFile', ...
                   'controlFile', 'wtbControlFile'};
    for i = 1:numel(input_files)
        path = context.(input_files{i});
        if ~isfile(path)
            error('WannierBSE:TBMissingInput', 'Required input not found: %s', path);
        end
    end
    if context.exchangeInteraction && ~isfile(context.wannierWinFile)
        error('WannierBSE:TBMissingInput', ...
              'Exchange interaction requires Wannier90 lattice input: %s', context.wannierWinFile);
    end
    if ~context.exchangeInteraction && ~isfile(context.wannierWinFile) && ~isfile(context.structureFile)
        error('WannierBSE:TBMissingInput', ...
              'Required lattice input not found. Expected either %s or %s.', ...
              context.wannierWinFile, context.structureFile);
    end
end

function ensure_dir(path)
    % Create output folders lazily so callers can provide fresh cache paths.
    if ~exist(path, 'dir')
        mkdir(path);
    end
end

function value = read_optional_scalar(path, key, fallback)
    % Read optional numerical controls without requiring older input files
    % to define every selector tolerance explicitly.
    value = fallback;
    if ~isfile(path)
        return;
    end
    txt = fileread(path);
    tok = regexp(txt, [key, '\s*=\s*([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)'], ...
                 'tokens', 'once');
    if ~isempty(tok)
        value = str2double(strrep(tok{1}, 'D', 'E'));
    end
end

function k = read_wbse_kmesh(filePath)
    % WBSE kmesh files carry five topology lines followed by Cartesian k
    % coordinates; only the first three columns are used by the TB solver.
    data = readmatrix(filePath, 'FileType', 'text', 'NumHeaderLines', 5);
    if size(data, 2) < 3 || isempty(data)
        error('WannierBSE:TBInvalidKmesh', ...
              'Expected at least three k-coordinate columns in %s.', filePath);
    end
    k = data(:, 1:3);
end

function PLV = read_lattice_vectors_from_win(filePath)
    % Extract the real-space primitive lattice vectors from the Wannier90
    % input file, preserving the convention used to phase the HR matrix.
    lines = readlines(filePath, 'EmptyLineRule', 'skip');
    startRow = find(contains(lower(lines), 'begin unit_cell_cart'), 1);
    endRow = find(contains(lower(lines), 'end unit_cell_cart'), 1);
    if isempty(startRow) || isempty(endRow) || endRow <= startRow
        error('WannierBSE:TBInvalidWin', ...
              'Cannot find unit_cell_cart block in %s.', filePath);
    end

    PLV = zeros(3, 3);
    found = 0;
    for iLine = startRow + 1:endRow - 1
        vals = sscanf(char(lines(iLine)), '%f').';
        if numel(vals) >= 3
            found = found + 1;
            PLV(found, :) = vals(1:3);
            if found == 3
                return
            end
        end
    end
    error('WannierBSE:TBInvalidWin', ...
          'Cannot parse three lattice vectors from %s.', filePath);
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: RESOLVE_LATTICE_VECTORS
%
% PURPOSE:
%   Internal helper used by TB_for_BSE and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function [PLV, latticeSource] = resolve_lattice_vectors(context)
    if isfile(context.wannierWinFile)
        latticeSource = context.wannierWinFile;
        PLV = read_lattice_vectors_from_win(latticeSource);
        return
    end
    latticeSource = context.structureFile;
    PLV = read_lattice_vectors_from_structure(latticeSource);
end

function PLV = read_lattice_vectors_from_structure(filePath)
    % Parse the same unit_cell_cart block accepted by User_input/structure.txt.
    lines = readlines(filePath, 'EmptyLineRule', 'skip');
    startRow = find(contains(lower(lines), 'begin unit_cell_cart'), 1);
    if isempty(startRow) || startRow + 3 > numel(lines)
        error('WannierBSE:TBInvalidStructure', ...
              'Cannot parse three lattice vectors from %s.', filePath);
    end

    PLV = zeros(3, 3);
    for iVector = 1:3
        vals = sscanf(char(lines(startRow + iVector)), '%f').';
        if numel(vals) < 3
            error('WannierBSE:TBInvalidStructure', ...
                  'Cannot parse lattice vector %d from %s.', iVector, filePath);
        end
        PLV(iVector, :) = vals(1:3);
    end
end

function [NW, BLVD, DEGN, HR] = read_wannier90_hr(filePath)
    % Parse wannier90_hr.dat into real-space lattice offsets, degeneracy
    % factors, and complex hopping matrices indexed by R, m, n.
    fid = fopen(filePath, 'r');
    if fid < 0
        error('WannierBSE:TBCannotOpenHR', 'Cannot open %s.', filePath);
    end
    cleanup = onCleanup(@() fclose(fid));

    fgetl(fid);
    NW = str2double(strtrim(fgetl(fid)));
    NBLV = str2double(strtrim(fgetl(fid)));
    if isnan(NW) || isnan(NBLV) || NW < 1 || NBLV < 1
        error('WannierBSE:TBInvalidHR', 'Cannot parse NW/NBLV from %s.', filePath);
    end

    DEGN = fscanf(fid, '%d', NBLV).';
    if numel(DEGN) ~= NBLV
        error('WannierBSE:TBInvalidHR', ...
              'Cannot parse %d degeneracy values from %s.', NBLV, filePath);
    end

    raw = fscanf(fid, '%f', [7, inf]).';
    expectedRows = NBLV * NW * NW;
    if size(raw, 1) ~= expectedRows
        error('WannierBSE:TBInvalidHR', ...
              'Expected %d HR rows in %s, but found %d.', ...
              expectedRows, filePath, size(raw, 1));
    end

    BLVD = zeros(NBLV, 3);
    HR = complex(zeros(NBLV, NW, NW));
    for iRow = 1:expectedRows
        iR = floor((iRow - 1) / (NW * NW)) + 1;
        iw = round(raw(iRow, 4));
        jw = round(raw(iRow, 5));
        if mod(iRow - 1, NW * NW) == 0
            BLVD(iR, :) = raw(iRow, 1:3);
        end
        HR(iR, iw, jw) = complex(raw(iRow, 6), raw(iRow, 7));
    end
end

function validate_inputs(k, NB, opts, reorder_map)
    % Fail early when band-edge indices, output counts, or basis maps are
    % inconsistent with the Hamiltonian dimension.
    if isempty(k)
        error('WannierBSE:TBNoKpoints', 'The WBSE kmesh has no k-points.');
    end
    if numel(reorder_map) ~= NB || any(sort(reorder_map(:)) ~= (1:NB).')
        error('WannierBSE:TBInvalidReorderMap', ...
              'WF_centers reorder map is not a permutation of 1:NB.');
    end
    if opts.VBM < 1 || opts.VBM >= NB
        error('WannierBSE:TBValenceWindow', 'VBM must be between 1 and NB - 1.');
    end
    if opts.CBM <= opts.VBM || opts.CBM > NB
        error('WannierBSE:TBConductionWindow', 'CBM must be between VBM + 1 and NB.');
    end
    if opts.outputNv < 1 || opts.outputNv > opts.selectNv
        error('WannierBSE:TBValenceWindow', 'Requested Nv exceeds the internally selected valence family.');
    end
    if opts.outputNc < 1 || opts.outputNc > opts.selectNc
        error('WannierBSE:TBConductionWindow', 'Requested Nc exceeds the internally selected conduction family.');
    end
end

function [E_data, C_data] = diagonalize_all_kpoints(k, BLVC, DEGN, HR, NB, Npar_TB)
    % Allocate the complete spectrum before the loop; every k-point stores
    % all NB energies and the full NB-by-NB eigenvector matrix.
    Nk = size(k, 1);
    E_data = zeros(NB, Nk);
    C_data = complex(zeros(NB, NB, Nk));
    use_parallel = Npar_TB > 1;
    if use_parallel
        parfor (ik = 1:Nk, Npar_TB)
            [E_data(:, ik), C_data(:, :, ik)] = diagonalize_one(k(ik, :), BLVC, DEGN, HR);
        end
    else
        for ik = 1:Nk
            [E_data(:, ik), C_data(:, :, ik)] = diagonalize_one(k(ik, :), BLVC, DEGN, HR);
        end
    end

end

function [energy, V] = diagonalize_one(kpt, BLVC, DEGN, HR)
    % Diagonalize one Bloch Hamiltonian and sort eigenpairs by ascending
    % energy so VBM/CBM indices are globally meaningful.
    Hk = build_hk(kpt, BLVC, DEGN, HR);
    [V, D] = eig(Hk, 'vector');
    [energy, order] = sort(real(D), 'ascend');
    V = V(:, order);
end

function Hk = build_hk(kpt, BLVC, DEGN, HR)
    % Fourier-transform the real-space Wannier Hamiltonian to the requested
    % k-point, including the Wannier90 degeneracy weights.
    phase = exp(1i * (BLVC * kpt(:))) ./ DEGN(:);
    weightedHR = HR .* reshape(phase, [], 1, 1);
    Hk = squeeze(sum(weightedHR, 1));
end

function spin_blocks = spin_blocks_after_reorder(spinor, reorder_map)
    % Rebuild the up/down index lists after applying the same basis
    % permutation used for the TB coefficients.
    NB = numel(spinor);
    reordered_spinor = strings(NB, 1);
    for old = 1:NB
        reordered_spinor(reorder_map(old)) = spinor(old);
    end
    spin_blocks = struct();
    spin_blocks.up = find(reordered_spinor == "up");
    spin_blocks.down = find(reordered_spinor == "down");
    if isempty(spin_blocks.up) || isempty(spin_blocks.down)
        error('WannierBSE:TBSpinBlocks', ...
              'WF_centers.txt must contain both up and down spinor labels.');
    end
end

function selection = select_output_window_by_band_index(E_data, C_data, opts, spin_blocks)
    % Choose the published BSE bands by fixed band-edge indices. This is the
    % key reproducibility rule: Nv/Nc changes only the number of rows written.
    Nk = size(E_data, 2);
    NB = size(E_data, 1);
    selection = struct();
    selection.V_idx = repmat((opts.VBM:-1:(opts.VBM - opts.outputNv + 1)).', 1, Nk);
    selection.C_idx = repmat((opts.CBM:(opts.CBM + opts.outputNc - 1)).', 1, Nk);
    selection.V_sz = zeros(opts.outputNv, Nk);
    selection.C_sz = zeros(opts.outputNc, Nk);
    selection.V_target_spin = zeros(opts.outputNv, Nk);
    selection.C_target_spin = zeros(opts.outputNc, Nk);
    selection.spin_paired = true(1, Nk);
    selection.spin_ambiguous = false(1, Nk);
    selection.overlap_used = false(1, Nk);
    selection.overlap_ref = zeros(1, Nk);
    selection.V_overlap = nan(opts.outputNv, Nk);
    selection.C_overlap = nan(opts.outputNc, Nk);
    selection.V_overlap_margin = nan(opts.outputNv, Nk);
    selection.C_overlap_margin = nan(opts.outputNc, Nk);
    selection.V_gap = zeros(opts.outputNv, Nk);
    selection.C_gap = zeros(opts.outputNc, Nk);
    selection.full_spectrum_sz = zeros(NB, Nk);
    for ik = 1:Nk
        for ib = 1:NB
            selection.full_spectrum_sz(ib, ik) = spin_expectation(C_data(:, ib, ik), spin_blocks);
        end
        selection.V_sz(:, ik) = selection.full_spectrum_sz(selection.V_idx(:, ik), ik);
        selection.C_sz(:, ik) = selection.full_spectrum_sz(selection.C_idx(:, ik), ik);
        selection.V_target_spin(:, ik) = sign(selection.V_sz(:, ik));
        selection.C_target_spin(:, ik) = sign(selection.C_sz(:, ik));
        selection.V_gap(:, ik) = local_band_gaps(E_data(:, ik), selection.V_idx(:, ik));
        selection.C_gap(:, ik) = local_band_gaps(E_data(:, ik), selection.C_idx(:, ik));
    end
end


function selection = enforce_conduction_spin_pairing(selection, E_data, opts)
    % Relabel conduction output states so c_i carries the same reliable spin
    % sign as v_i at each k-point. D3h and time-reversal stages run after
    % this pass and use C_target_spin to preserve the paired convention.
    nV = size(selection.V_idx, 1);
    nC = size(selection.C_idx, 1);
    nPair = min(nV, nC);
    if nPair == 0
        return
    end

    NB = size(E_data, 1);
    Nk = size(E_data, 2);
    pad = max(0, opts.d3h_candidate_padding);
    for ik = 1:Nk
        current = selection.C_idx(:, ik);
        first = opts.CBM;
        last = min(NB, max(current) + pad);
        candidates = first:last;

        targetSpin = sign(selection.C_sz(:, ik));
        for iPair = 1:nPair
            vSpin = selection.V_sz(iPair, ik);
            if abs(vSpin) >= opts.spin_threshold
                targetSpin(iPair) = sign(vSpin);
            else
                selection.spin_ambiguous(ik) = true;
            end
        end
        selection.C_target_spin(:, ik) = targetSpin;

        assignments = d3h_ordered_assignments(candidates, nC, ...
            selection.full_spectrum_sz(:, ik), targetSpin, opts);
        if isempty(assignments)
            selection.spin_paired(ik) = false;
            selection.spin_ambiguous(ik) = true;
            continue
        end

        bestCost = Inf;
        bestAssign = current;
        currentEnergy = E_data(current, ik);
        for iAssign = 1:size(assignments, 1)
            candidate = assignments(iAssign, :).';
            candidateEnergy = E_data(candidate, ik);
            cost = sum(abs(candidateEnergy - currentEnergy)) / 0.10;
            for iPair = 1:nPair
                wanted = targetSpin(iPair);
                if wanted ~= 0
                    sz = selection.full_spectrum_sz(candidate(iPair), ik);
                    if abs(sz) >= opts.spin_threshold
                        cost = cost + 100.0 * abs(sign(sz) - wanted);
                    end
                end
            end
            if cost < bestCost
                bestCost = cost;
                bestAssign = candidate;
            end
        end

        selection.C_idx(:, ik) = bestAssign;
        selection.C_sz(:, ik) = selection.full_spectrum_sz(bestAssign, ik);
        selection.C_gap(:, ik) = local_band_gaps(E_data(:, ik), bestAssign);
        for iPair = 1:nPair
            vSpin = selection.V_sz(iPair, ik);
            cSpin = selection.C_sz(iPair, ik);
            if abs(vSpin) >= opts.spin_threshold && abs(cSpin) >= opts.spin_threshold && sign(vSpin) ~= sign(cSpin)
                selection.spin_paired(ik) = false;
            end
        end
    end
end

function sz = spin_expectation(coeff, spin_blocks)
    % Compute normalized spin polarization in the reordered WBSE basis using
    % orbital weights in the up and down spin blocks.
    weight = abs(coeff) .^ 2;
    total = sum(weight);
    if total <= 0
        sz = NaN;
        return
    end
    w_up = sum(weight(spin_blocks.up)) / total;
    w_down = sum(weight(spin_blocks.down)) / total;
    sz = w_up - w_down;
end


function gaps = local_band_gaps(E, idx)
    % Record the nearest neighboring energy separation for each selected
    % band; small values mark nearly degenerate local manifolds.
    NB = numel(E);
    gaps = zeros(numel(idx), 1);
    for i = 1:numel(idx)
        b = idx(i);
        left = inf;
        right = inf;
        if b > 1
            left = abs(E(b) - E(b - 1));
        end
        if b < NB
            right = abs(E(b + 1) - E(b));
        end
        gaps(i) = min(left, right);
    end
end

function write_selected_band_files(outputDir, k, E_data, C_data, selection, opts)
    % Keep the TB cache self-consistent with the current Nv/Nc request by
    % removing stale band files before writing the active output window.
    cleanup_stale_band_files(outputDir);
    for n = 1:opts.outputNv
        fname = fullfile(outputDir, sprintf('v%d_TB_WBSE.txt', n));
        write_one_selected_band_file(fname, k, E_data, C_data, selection.V_idx(n, :));
    end
    for n = 1:opts.outputNc
        fname = fullfile(outputDir, sprintf('c%d_TB_WBSE.txt', n));
        write_one_selected_band_file(fname, k, E_data, C_data, selection.C_idx(n, :));
    end
end

function cleanup_stale_band_files(outputDir)
    % Prevent old larger-basis outputs from being mistaken as current inputs
    % after the user reduces Nv or Nc.
    stale = [dir(fullfile(outputDir, 'v*_TB_WBSE.txt')); ...
             dir(fullfile(outputDir, 'c*_TB_WBSE.txt'))];
    for iFile = 1:numel(stale)
        delete(fullfile(outputDir, stale(iFile).name));
    end
end

function write_one_selected_band_file(filename, k, E_data, C_data, selectedBands)
    % Write one WBSE band file as repeated k-point blocks: k vector, scalar
    % energy, then the full complex coefficient vector in WBSE basis order.
    fid = fopen(filename, 'w');
    if fid == -1
        error('WannierBSE:TBTBFileWrite', 'Unable to write %s.', filename);
    end
    cleanup = onCleanup(@() fclose(fid));

    Nk = size(k, 1);
    NB = size(C_data, 1);
    for ik = 1:Nk
        band = selectedBands(ik);
        coeff = unify_global_phase(C_data(:, band, ik));
        block = [sprintf('%20.16f %20.16f %20.16f\n', k(ik, 1), k(ik, 2), k(ik, 3)), ...
                 sprintf('%20.16f\n', E_data(band, ik))];
        for ib = 1:NB
            val = coeff(ib);
            block = [block, sprintf('%20.16f %20.16f %20.16f\n', real(val), imag(val), abs(val))]; %#ok<AGROW>
        end
        fprintf(fid, '%s', block);
    end
end

function C_data = spin_adapt_exact_degenerate_clusters(E_data, C_data, spin_blocks, opts)
    % Within exactly degenerate eigenspaces, choose a deterministic basis by
    % diagonalizing Sz projected into the degenerate subspace.
    [NB, Nk] = size(E_data);
    Sz = zeros(NB, 1);
    Sz(spin_blocks.up) = 1;
    Sz(spin_blocks.down) = -1;
    for ik = 1:Nk
        first = 1;
        while first <= NB
            last = first;
            while last < NB && abs(E_data(last + 1, ik) - E_data(last, ik)) <= opts.exact_rotation_gap_eV
                last = last + 1;
            end
            if last > first
                bands = first:last;
                U = C_data(:, bands, ik);
                [Q, ~] = qr(U, 0);
                projectedSpin = (Q' * (Sz .* Q) + (Sz .* Q)' * Q) / 2;
                [rotation, values] = eig(projectedSpin, 'vector');
                [~, order] = sort(real(values), 'descend');
                adapted = Q * rotation(:, order);
                C_data(:, bands, ik) = adapted;
            end
            first = last + 1;
        end
    end
end


function selection = enforce_d3h_energy_orbit_consistency(selection, E_data, k, opts)
    % Apply conservative C3 orbit relabeling only for small output windows,
    % where exhaustive assignment is tractable and avoids arbitrary swaps.
    maps = find_d3h_partners(k, opts);
    orbits = build_symmetry_orbits(maps);
    for iOrbit = 1:numel(orbits)
        orbit = orbits{iOrbit};
        if numel(orbit) < 2
            continue
        end
        selection = relabel_d3h_family(selection, E_data, orbit, 'V', opts);
        selection = relabel_d3h_family(selection, E_data, orbit, 'C', opts);
    end
end

function selection = relabel_d3h_family(selection, E_data, orbit, family, opts)
    % For one valence or conduction family, search assignments that reduce
    % energy spread over a C3 orbit without large local energy shifts.
    indexField = [family, '_idx'];
    spinField = [family, '_sz'];
    targetField = [family, '_target_spin'];
    gapField = [family, '_gap'];
    nLabel = size(selection.(indexField), 1);
    if nLabel == 0
        return
    end
    if nLabel > 4
        return
    end

    candidates = d3h_orbit_candidate_window(selection, E_data, orbit, family, opts);
    reference = orbit(1);
    orbitAssignments = cell(numel(orbit), 1);
    preparedCosts = cell(numel(orbit), 1);
    for iK = 1:numel(orbit)
        ik = orbit(iK);
        orbitAssignments{iK} = d3h_ordered_assignments(candidates, nLabel, ...
            selection.full_spectrum_sz(:, ik), selection.(targetField)(:, ik), opts);
        if iK > 1 && ~isempty(orbitAssignments{iK})
            preparedCosts{iK} = prepare_d3h_assignment_cost_data( ...
                selection, E_data, ik, family, orbitAssignments{iK}, opts);
        end
    end
    referenceAssignments = orbitAssignments{1};
    if isempty(referenceAssignments)
        return
    end

    bestCost = Inf;
    bestByOrbit = cell(numel(orbit), 1);
    for iRef = 1:size(referenceAssignments, 1)
        trial = cell(numel(orbit), 1);
        trial{1} = referenceAssignments(iRef, :).';
        referenceEnergy = E_data(trial{1}, reference);
        cost = d3h_assignment_regularization(selection, E_data, reference, family, trial{1}, opts);
        for iK = 2:numel(orbit)
            ik = orbit(iK);
            assignments = orbitAssignments{iK};
            if isempty(assignments)
                cost = Inf;
                break
            end
            [localAssign, localCost] = best_d3h_assignment( ...
                selection, E_data, ik, family, assignments, referenceEnergy, opts, preparedCosts{iK});
            trial{iK} = localAssign;
            cost = cost + localCost;
        end
        if cost < bestCost
            bestCost = cost;
            bestByOrbit = trial;
        end
    end

    if ~isfinite(bestCost)
        return
    end

    currentByOrbit = cell(numel(orbit), 1);
    for iK = 1:numel(orbit)
        currentByOrbit{iK} = selection.(indexField)(:, orbit(iK));
    end
    currentSpread = d3h_orbit_energy_spread(E_data, orbit, currentByOrbit);
    trialSpread = d3h_orbit_energy_spread(E_data, orbit, bestByOrbit);
    maxShift = d3h_max_assignment_shift(E_data, orbit, currentByOrbit, bestByOrbit);
    if trialSpread >= currentSpread - opts.d3h_energy_tolerance_eV || maxShift > opts.d3h_max_energy_shift_eV
        return
    end

    for iK = 1:numel(orbit)
        ik = orbit(iK);
        selection.(indexField)(:, ik) = bestByOrbit{iK};
        selection.(spinField)(:, ik) = selection.full_spectrum_sz(bestByOrbit{iK}, ik);
        selection.(gapField)(:, ik) = local_band_gaps(E_data(:, ik), bestByOrbit{iK});
        if isfield(selection, [family, '_overlap'])
            selection.([family, '_overlap'])(:, ik) = NaN;
            selection.([family, '_overlap_margin'])(:, ik) = NaN;
        end
    end
end



function spread = d3h_orbit_energy_spread(E_data, orbit, assignments)
    % Measure the largest same-label energy spread within a symmetry orbit.
    nLabel = numel(assignments{1});
    spread = 0;
    for iLabel = 1:nLabel
        energies = zeros(numel(orbit), 1);
        for iK = 1:numel(orbit)
            energies(iK) = E_data(assignments{iK}(iLabel), orbit(iK));
        end
        spread = max(spread, max(energies) - min(energies));
    end
end

function maxShift = d3h_max_assignment_shift(E_data, orbit, currentByOrbit, trialByOrbit)
    % Bound how far a proposed D3h relabel would move any selected state in
    % energy, preventing jumps to a different physical manifold.
    maxShift = 0;
    for iK = 1:numel(orbit)
        ik = orbit(iK);
        currentEnergy = E_data(currentByOrbit{iK}, ik);
        trialEnergy = E_data(trialByOrbit{iK}, ik);
        maxShift = max(maxShift, max(abs(trialEnergy - currentEnergy)));
    end
end

function candidates = d3h_orbit_candidate_window(selection, E_data, orbit, family, opts)
    % Build a compact candidate band-index range around the states already
    % selected on the symmetry orbit.
    indexField = [family, '_idx'];
    NB = size(E_data, 1);
    pad = max(1, opts.d3h_candidate_padding);
    selected = selection.(indexField)(:, orbit);
    first = max(1, min(selected(:)) - pad);
    last = min(NB, max(selected(:)) + pad);
    candidates = first:last;
end

function candidates = d3h_candidate_window(E_data, family, opts)
    % Provide a broader valence/conduction search region for time-reversal
    % partner matching while respecting the VBM/CBM separation.
    NB = size(E_data, 1);
    pad = max(0, opts.d3h_candidate_padding);
    if strcmp(family, 'V')
        first = 1;
        last = min(NB, opts.VBM + pad);
    else
        first = max(1, opts.CBM - pad);
        last = NB;
    end
    candidates = first:last;
end

function assignments = d3h_ordered_assignments(candidates, nLabel, szAll, targetSpin, opts)
    % Build all non-repeating label assignments compatible with the target
    % spin signs whenever the spin expectation is reliable.
    filtered = cell(nLabel, 1);
    for iLabel = 1:nLabel
        wanted = sign(targetSpin(iLabel));
        pool = candidates(:);
        if wanted ~= 0
            spinPool = pool(sign(szAll(pool)) == wanted & abs(szAll(pool)) >= opts.spin_threshold);
            if isempty(spinPool)
                spinPool = pool(sign(szAll(pool)) == wanted);
            end
            if isempty(spinPool)
                assignments = zeros(0, nLabel);
                return
            end
            pool = spinPool;
        end
        filtered{iLabel} = pool(:).';
    end
    assignments = build_assignments_recursive(filtered, nLabel, [], []);
end

function assignments = build_assignments_recursive(filtered, nLabel, current, used)
    % Recursively enumerate small assignment spaces for exact relabel costs.
    if numel(current) == nLabel
        assignments = current;
        return
    end
    iLabel = numel(current) + 1;
    assignments = zeros(0, nLabel);
    for candidate = filtered{iLabel}
        if any(used == candidate)
            continue
        end
        sub = build_assignments_recursive(filtered, nLabel, [current, candidate], [used, candidate]);
        assignments = [assignments; sub]; %#ok<AGROW>
    end
end

function prepared = prepare_d3h_assignment_cost_data(selection, E_data, ik, family, assignments, opts)
    % Cache assignment-invariant costs for one orbit member. The formulas
    % intentionally call the scalar regularization routine once per
    % assignment to preserve existing arithmetic semantics.
    nAssign = size(assignments, 1);
    prepared.candidateEnergy = cell(nAssign, 1);
    prepared.regularization = zeros(nAssign, 1);
    for iAssign = 1:nAssign
        candidate = assignments(iAssign, :).';
        prepared.candidateEnergy{iAssign} = E_data(candidate, ik);
        prepared.regularization(iAssign) = d3h_assignment_regularization( ...
            selection, E_data, ik, family, candidate, opts);
    end
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: BEST_D3H_ASSIGNMENT
%
% PURPOSE:
%   Internal helper used by TB_for_BSE and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function [bestAssign, bestCost] = best_d3h_assignment( ...
    selection, E_data, ik, family, assignments, referenceEnergy, opts, prepared)
    % Choose the assignment that best matches the reference-orbit energy set
    % while retaining regularization against unnecessary relabels.
    if nargin < 8 || isempty(prepared)
        prepared = prepare_d3h_assignment_cost_data(selection, E_data, ik, family, assignments, opts);
    end
    bestCost = Inf;
    bestAssign = assignments(1, :).';
    for iAssign = 1:size(assignments, 1)
        candidate = assignments(iAssign, :).';
        energyCost = sum(abs(prepared.candidateEnergy{iAssign} - referenceEnergy)) / max(opts.d3h_energy_tolerance_eV, 1.0e-12);
        regCost = prepared.regularization(iAssign);
        cost = energyCost + regCost;
        if cost < bestCost
            bestCost = cost;
            bestAssign = candidate;
        end
    end
end

function cost = d3h_assignment_regularization(selection, E_data, ik, family, candidate, opts)
    % Penalize assignments that move far from the current fixed-window band
    % identity or contradict reliable spin signs.
    indexField = [family, '_idx'];
    targetField = [family, '_target_spin'];
    current = selection.(indexField)(:, ik);
    spinTarget = selection.(targetField)(:, ik);
    currentEnergy = E_data(current, ik);
    candidateEnergy = E_data(candidate, ik);
    energyShift = sum(abs(candidateEnergy - currentEnergy)) / 0.10;
    spinPenalty = 0;
    for iLabel = 1:numel(candidate)
        wanted = sign(spinTarget(iLabel));
        if wanted ~= 0
            sz = selection.full_spectrum_sz(candidate(iLabel), ik);
            if abs(sz) >= opts.spin_threshold
                spinPenalty = spinPenalty + 100.0 * abs(sign(sz) - wanted);
            end
        end
    end
    cost = energyShift + spinPenalty;
end

function maps = find_d3h_partners(k, opts)
    % Map each k-point to its C3-rotated partners using reciprocal-lattice
    % equivalence, so boundary points are matched modulo G vectors.
    Nk = size(k, 1);
    maps = zeros(Nk, 3);
    ops = d3h_in_plane_operations();
    b1 = opts.reciprocal(1, 1:2);
    b2 = opts.reciprocal(2, 1:2);
    reciprocal = [b1(:), b2(:)];
    lookup = build_d3h_k_lookup(k(:, 1:2), reciprocal, opts.d3h_k_tolerance);
    for iOp = 1:numel(ops)
        op = ops{iOp};
        for ik = 1:Nk
            target = (op * k(ik, 1:2).').';
            [maps(ik, iOp), residual] = indexed_nearest_k_mod_reciprocal( ...
                k(:, 1:2), target, reciprocal, lookup);
            if residual > opts.d3h_k_tolerance
                maps(ik, iOp) = 0;
            end
        end
    end
end

function ops = d3h_in_plane_operations()
    % In the 2D k plane, the relevant D3h spatial orbit is generated by the
    % three C3 rotations; horizontal reflection does not change in-plane k.
    ops = cell(1, 3);
    for n = 0:2
        angle = 2 * pi * n / 3;
        ops{n + 1} = [cos(angle), -sin(angle); sin(angle), cos(angle)];
    end
end

function [bestIndex, bestResidual] = nearest_k_mod_reciprocal(kxy, target, reciprocal)
    % Compare k-points after folding the displacement into the first
    % reciprocal-lattice cell.
    delta = kxy - target;
    fractional = (reciprocal \ delta.').';
    fractional = fractional - round(fractional);
    residualCartesian = (reciprocal * fractional.').';
    [bestResidual, bestIndex] = min(sqrt(sum(residualCartesian .^ 2, 2)));
end

function lookup = build_d3h_k_lookup(kxy, reciprocal, tolerance)
    % Index actual k-points and nearby reciprocal images; residual checks
    % below remain authoritative for periodic equivalence.
    Nk = size(kxy, 1);
    translations = zeros(9, 2);
    row = 0;
    for n1 = -1:1
        for n2 = -1:1
            row = row + 1;
            translations(row, :) = [n1, n2];
        end
    end

    imageCoordinates = zeros(9 * Nk, 2);
    imageLabels = zeros(9 * Nk, 1);
    row = 0;
    b1 = reciprocal(:, 1).';
    b2 = reciprocal(:, 2).';
    for iTranslation = 1:size(translations, 1)
        rows = row + (1:Nk);
        imageCoordinates(rows, :) = kxy ...
            - translations(iTranslation, 1) * b1 ...
            - translations(iTranslation, 2) * b2;
        imageLabels(rows) = (1:Nk).';
        row = row + Nk;
    end

    imageKeys = round(imageCoordinates / tolerance);
    [sortedImageKeys, imageOrder] = sortrows(imageKeys, [1, 2]);
    sortedImageLabels = imageLabels(imageOrder);
    [uniqueImageKeys, firstRows] = unique(sortedImageKeys, 'rows', 'first');
    [~, lastRows] = unique(sortedImageKeys, 'rows', 'last');

    keyRangeLookup = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for iKey = 1:size(uniqueImageKeys, 1)
        keyRangeLookup(integer_key_string_2d(uniqueImageKeys(iKey, :))) = ...
            [firstRows(iKey), lastRows(iKey)];
    end

    [offset1, offset2] = ndgrid(-1:1, -1:1);
    lookup.keyRangeLookup = keyRangeLookup;
    lookup.sortedImageLabels = sortedImageLabels;
    lookup.queryKeyOffsets = [offset1(:), offset2(:)];
    lookup.tolerance = tolerance;
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: INDEXED_NEAREST_K_MOD_RECIPROCAL
%
% PURPOSE:
%   Internal helper used by TB_for_BSE and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function [bestIndex, bestResidual] = indexed_nearest_k_mod_reciprocal(kxy, target, reciprocal, lookup)
    query = round(target / lookup.tolerance);
    queryKeys = query + lookup.queryKeyOffsets;
    candidateIndices = lookup_d3h_candidate_indices( ...
        lookup.keyRangeLookup, lookup.sortedImageLabels, queryKeys);
    if isempty(candidateIndices)
        [bestIndex, bestResidual] = nearest_k_mod_reciprocal(kxy, target, reciprocal);
        return
    end

    [localIndex, bestResidual] = nearest_k_mod_reciprocal_subset( ...
        kxy, target, reciprocal, candidateIndices);
    bestIndex = candidateIndices(localIndex);
    if bestResidual > lookup.tolerance
        [bestIndex, bestResidual] = nearest_k_mod_reciprocal(kxy, target, reciprocal);
    end
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: LOOKUP_D3H_CANDIDATE_INDICES
%
% PURPOSE:
%   Internal helper used by TB_for_BSE and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function candidateIndices = lookup_d3h_candidate_indices(keyRangeLookup, sortedImageLabels, queryKeys)
    candidateIndices = zeros(0, 1);
    for iKey = 1:size(queryKeys, 1)
        key = integer_key_string_2d(queryKeys(iKey, :));
        if ~isKey(keyRangeLookup, key)
            continue
        end
        rows = keyRangeLookup(key);
        candidateIndices = [candidateIndices; sortedImageLabels(rows(1):rows(2))]; %#ok<AGROW>
    end
    if numel(candidateIndices) > 1
        candidateIndices = unique(candidateIndices);
    end
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: NEAREST_K_MOD_RECIPROCAL_SUBSET
%
% PURPOSE:
%   Internal helper used by TB_for_BSE and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function [bestLocalIndex, bestResidual] = nearest_k_mod_reciprocal_subset(kxy, target, reciprocal, candidateIndices)
    delta = kxy(candidateIndices, :) - target;
    fractional = (reciprocal \ delta.').';
    fractional = fractional - round(fractional);
    residualCartesian = (reciprocal * fractional.').';
    [bestResidual, bestLocalIndex] = min(sqrt(sum(residualCartesian .^ 2, 2)));
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: INTEGER_KEY_STRING_2D
%
% PURPOSE:
%   Internal helper used by TB_for_BSE and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function key = integer_key_string_2d(keyValues)
    key = sprintf('%.0f,%.0f', keyValues(1), keyValues(2));
end

function orbits = build_symmetry_orbits(maps)
    % Merge pairwise symmetry links into closed k-point orbits.
    Nk = size(maps, 1);
    visited = false(Nk, 1);
    orbits = {};
    for ik = 1:Nk
        if visited(ik)
            continue
        end
        orbit = unique(nonzeros(maps(ik, :))).';
        orbit = unique([ik, orbit]);
        changed = true;
        while changed
            changed = false;
            for member = orbit
                linked = unique(nonzeros(maps(member, :))).';
                merged = unique([orbit, linked]);
                if numel(merged) ~= numel(orbit)
                    orbit = merged;
                    changed = true;
                end
            end
        end
        visited(orbit) = true;
        orbits{end + 1} = orbit; %#ok<AGROW>
    end
end

function selection = enforce_time_reversal_partner_states(selection, E_data, C_data, k, opts, partner)
    % For each non-self TR pair, choose the partner states that match energy
    % and, when reliable, opposite spin within the selected family.
    selection.TR_partner = partner(:).';
    if ~isfield(selection, 'TR_relabel_applied')
        selection.TR_relabel_applied = false(1, size(k, 1));
    end
    for ik = 1:numel(partner)
        jk = partner(ik);
        if jk == 0 || jk == ik || ik > jk
            continue
        end
        [selection, changedV] = match_time_reversal_family( ...
            selection, E_data, C_data, ik, jk, 'V', d3h_candidate_window(E_data, 'V', opts), opts);
        [selection, changedC] = match_time_reversal_family( ...
            selection, E_data, C_data, ik, jk, 'C', d3h_candidate_window(E_data, 'C', opts), opts);
        selection.TR_relabel_applied(jk) = changedV || changedC;
    end
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: MATCH_TIME_REVERSAL_FAMILY
%
% PURPOSE:
%   Internal helper used by TB_for_BSE and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function [selection, changed] = match_time_reversal_family( ...
        selection, E_data, C_data, ik, jk, family, candidates, opts)
    % Match one valence or conduction output family at k and -k using a
    % cost hierarchy: energy, reliable spin reversal, then TR-overlap gauge.
    indexField = [family, '_idx'];
    spinField = [family, '_sz'];
    gapField = [family, '_gap'];
    sourceIdx = selection.(indexField)(:, ik);
    originalIdx = selection.(indexField)(:, jk);
    targetIdx = zeros(size(sourceIdx));
    used = false(size(E_data, 1), 1);
    energyScale = max(opts.tr_energy_tolerance_eV, 1.0e-12);

    for ib = 1:numel(sourceIdx)
        available = candidates(~used(candidates));
        available = available(:);
        sourceEnergy = E_data(sourceIdx(ib), ik);
        sourceSpin = selection.(spinField)(ib, ik);
        targetSpin = selection.full_spectrum_sz(available, jk);
        energyError = abs(sourceEnergy - E_data(available, jk));
        if abs(sourceSpin) >= opts.spin_threshold
            reliable = abs(targetSpin) >= opts.spin_threshold;
            opposite = reliable & sign(targetSpin) == -sign(sourceSpin);
            energyWindow = max(opts.tr_energy_tolerance_eV, opts.overlap_degeneracy_gap_eV);
            preferred = opposite & energyError <= energyWindow;
            if any(preferred)
                available = available(preferred);
                targetSpin = targetSpin(preferred);
                energyError = energyError(preferred);
            end
        end
        cost = energyError / energyScale;
        if abs(sourceSpin) >= opts.spin_threshold
            reliable = abs(targetSpin) >= opts.spin_threshold;
            cost(reliable) = cost(reliable) + abs(sourceSpin + targetSpin(reliable));
        end
        targetStates = C_data(:, available, jk);
        thetaSource = theta_spin_block(C_data(:, sourceIdx(ib), ik));
        overlapCost = 1 - abs(thetaSource' * targetStates)'.^2;
        cost = cost + 1.0e-3 * overlapCost;
        [~, local] = min(cost);
        targetIdx(ib) = available(local);
        used(targetIdx(ib)) = true;
    end

    changed = any(targetIdx ~= originalIdx);
    selection.(indexField)(:, jk) = targetIdx;
    selection.(spinField)(:, jk) = selection.full_spectrum_sz(targetIdx, jk);
    selection.(gapField)(:, jk) = local_band_gaps(E_data(:, jk), targetIdx);
    selection.([family, '_target_spin'])(:, jk) = -sign(selection.(spinField)(:, ik));
    if isfield(selection, [family, '_overlap'])
        selection.([family, '_overlap'])(:, jk) = NaN;
        selection.([family, '_overlap_margin'])(:, jk) = NaN;
    end
end

function selection = relabel_time_reversal_pairs(selection, E_data, k, opts, partner)
    % Reorder labels within each selected TR pair so file labels are as
    % consistent as possible between k and -k.
    selection.TR_partner = partner(:).';
    selection.TR_relabel_applied = false(1, size(k, 1));
    for ik = 1:numel(partner)
        jk = partner(ik);
        if jk == 0 || jk == ik || ik > jk
            continue
        end
        [selection, changedV] = relabel_selected_family(selection, E_data, ik, jk, 'V', opts);
        [selection, changedC] = relabel_selected_family(selection, E_data, ik, jk, 'C', opts);
        selection.TR_relabel_applied(jk) = changedV || changedC;
    end
end

function partner = find_time_reversal_partners(k, opts)
    % Locate the index j satisfying k_i + k_j = G within tolerance.
    % Integerized image keys reduce the lookup to a small candidate bucket;
    % the original Cartesian residual test remains the acceptance criterion.
    Nk = size(k, 1);
    partner = zeros(Nk, 1);
    b1 = opts.reciprocal(1, :);
    b2 = opts.reciprocal(2, :);
    keyScale = opts.tr_k_tolerance;
    translations = zeros(9, 3);
    row = 0;
    for n1 = -1:1
        for n2 = -1:1
            row = row + 1;
            translations(row, :) = [n1, n2, 0];
        end
    end

    imageCoordinates = zeros(9 * Nk, 3);
    imageLabels = zeros(9 * Nk, 2);
    row = 0;
    for iTranslation = 1:size(translations, 1)
        rows = row + (1:Nk);
        imageCoordinates(rows, :) = k - translations(iTranslation, 1) * b1 ...
            - translations(iTranslation, 2) * b2;
        imageLabels(rows, :) = [ (1:Nk).', repmat(iTranslation, Nk, 1) ];
        row = row + Nk;
    end
    imageKeys = round(imageCoordinates / keyScale);
    queryOffsets = -1:1;
    [offset1, offset2, offset3] = ndgrid(queryOffsets, queryOffsets, queryOffsets);
    queryKeyOffsets = [offset1(:), offset2(:), offset3(:)];
    [sortedImageKeys, imageOrder] = sortrows(imageKeys, [1, 2, 3]);
    sortedImageLabels = imageLabels(imageOrder, :);
    [uniqueImageKeys, firstRows] = unique(sortedImageKeys, 'rows', 'first');
    [~, lastRows] = unique(sortedImageKeys, 'rows', 'last');
    keyRangeLookup = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for iKey = 1:size(uniqueImageKeys, 1)
        keyRangeLookup(integer_key_string(uniqueImageKeys(iKey, :))) = [firstRows(iKey), lastRows(iKey)];
    end

    for ik = 1:Nk
        query = round((-k(ik, :)) / keyScale);
        queryKeys = query + queryKeyOffsets;
        candidateLabels = lookup_time_reversal_candidates( ...
            keyRangeLookup, sortedImageLabels, queryKeys);
        if isempty(candidateLabels)
            partner(ik) = find_time_reversal_partner_reference(k, ik, b1, b2, opts.tr_k_tolerance);
            continue
        end

        candidateTranslations = translations(candidateLabels(:, 2), :);
        delta = k(ik, :) + k(candidateLabels(:, 1), :) ...
            - candidateTranslations(:, 1) .* b1 ...
            - candidateTranslations(:, 2) .* b2;
        residual = sqrt(sum(delta.^2, 2));
        candidateOrder = sortrows([candidateLabels, residual], [1, 2, 3]);
        bestResidual = candidateOrder(1, 3);
        bestIndex = candidateOrder(1, 1);
        if bestResidual <= opts.tr_k_tolerance
            partner(ik) = bestIndex;
        else
            partner(ik) = find_time_reversal_partner_reference(k, ik, b1, b2, opts.tr_k_tolerance);
        end
    end
end

function candidateLabels = lookup_time_reversal_candidates(keyRangeLookup, sortedImageLabels, queryKeys)
    % Use the same integer keys as the original row search, but retrieve
    % matching image buckets directly before residual verification.
    candidateLabels = zeros(0, 2);
    for iKey = 1:size(queryKeys, 1)
        key = integer_key_string(queryKeys(iKey, :));
        if ~isKey(keyRangeLookup, key)
            continue
        end
        rows = keyRangeLookup(key);
        candidateLabels = [candidateLabels; sortedImageLabels(rows(1):rows(2), :)]; %#ok<AGROW>
    end
    if size(candidateLabels, 1) > 1
        candidateLabels = unique(candidateLabels, 'rows');
    end
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: INTEGER_KEY_STRING
%
% PURPOSE:
%   Internal helper used by TB_for_BSE and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function key = integer_key_string(keyValues)
    key = sprintf('%.0f,%.0f,%.0f', keyValues(1), keyValues(2), keyValues(3));
end

function bestIndex = find_time_reversal_partner_reference(k, ik, b1, b2, tolerance)
    % Exact reference fallback for a key bucket boundary or unusual mesh.
    Nk = size(k, 1);
    best = inf;
    bestIndex = 0;
    for jk = 1:Nk
        base = k(ik, :) + k(jk, :);
        for n1 = -1:1
            for n2 = -1:1
                residual = norm(base - n1 * b1 - n2 * b2);
                if residual < best
                    best = residual;
                    bestIndex = jk;
                end
            end
        end
    end
    if best > tolerance
        bestIndex = 0;
    end
end

function [selection, changed] = relabel_selected_family(selection, E_data, ik, jk, family, opts)
    % Apply the lowest-cost TR permutation to all per-label selection fields.
    indexField = [family, '_idx'];
    spinField = [family, '_sz'];
    sourceIdx = selection.(indexField)(:, ik);
    targetIdx = selection.(indexField)(:, jk);
    fields = {[family, '_idx'], [family, '_sz'], [family, '_target_spin'], ...
              [family, '_overlap'], [family, '_overlap_margin'], [family, '_gap']};
    if numel(sourceIdx) > 7
        if strcmp(family, 'V')
            nRelabel = opts.outputNv;
        else
            nRelabel = opts.outputNc;
        end
        rows = 1:nRelabel;
        permutation = best_tr_permutation(E_data(sourceIdx(rows), ik), E_data(targetIdx(rows), jk), ...
                                          selection.(spinField)(rows, ik), selection.(spinField)(rows, jk), opts);
        changed = any(permutation ~= rows.');
        for iField = 1:numel(fields)
            name = fields{iField};
            if isfield(selection, name)
                values = selection.(name)(rows, jk);
                selection.(name)(rows, jk) = values(permutation);
            end
        end
        return
    end
    permutation = best_tr_permutation(E_data(sourceIdx, ik), E_data(targetIdx, jk), ...
                                      selection.(spinField)(:, ik), selection.(spinField)(:, jk), opts);
    changed = any(permutation ~= (1:numel(permutation)).');
    for iField = 1:numel(fields)
        name = fields{iField};
        if isfield(selection, name)
            values = selection.(name)(:, jk);
            selection.(name)(:, jk) = values(permutation);
        end
    end
end

function permutation = best_tr_permutation(Eleft, Eright, Szleft, Szright, opts)
    % Use exact permutations for small output windows; fall back to greedy
    % matching only when the label count is too large for factorial search.
    n = numel(Eleft);
    energyScale = max(opts.tr_energy_tolerance_eV, 1.0e-12);
    if n <= 7
        candidates = perms(1:n);
        bestCost = NaN;
        permutation = (1:n).';
        for row = 1:size(candidates, 1)
            candidate = candidates(row, :).';
            cost = tr_permutation_cost(Eleft, Eright, Szleft, Szright, candidate, opts, energyScale);
            if isnan(bestCost) || cost < bestCost - 10 * eps(max(1, abs(bestCost)))
                bestCost = cost;
                permutation = candidate;
            end
        end
        return
    end

    permutation = greedy_tr_permutation(Eleft, Eright, Szleft, Szright, opts, energyScale);
end

function cost = tr_permutation_cost(Eleft, Eright, Szleft, Szright, candidate, opts, energyScale)
    % Combine energy mismatch with reliable spin-reversal mismatch for one
    % proposed TR label permutation.
    cost = sum(abs(Eleft - Eright(candidate))) / energyScale;
    for i = 1:numel(candidate)
        if abs(Szleft(i)) >= opts.spin_threshold && abs(Szright(candidate(i))) >= opts.spin_threshold
            cost = cost + abs(Szleft(i) + Szright(candidate(i)));
        end
    end
end

function permutation = greedy_tr_permutation(Eleft, Eright, Szleft, Szright, opts, energyScale)
    % Assign large label sets sequentially to avoid factorial memory and
    % runtime growth.
    n = numel(Eleft);
    permutation = zeros(n, 1);
    unused = true(n, 1);
    for i = 1:n
        available = find(unused);
        cost = abs(Eleft(i) - Eright(available)) / energyScale;
        reliable = abs(Szleft(i)) >= opts.spin_threshold & abs(Szright(available)) >= opts.spin_threshold;
        cost(reliable) = cost(reliable) + abs(Szleft(i) + Szright(available(reliable)));
        [~, local] = min(cost);
        permutation(i) = available(local);
        unused(permutation(i)) = false;
    end
end

function C_data = align_time_reversal_gauge(C_data, E_data, selection, opts)
    % Fix phases and allowed exact-degenerate rotations for selected TR
    % partners before coefficient files are written.
    partner = selection.TR_partner(:);
    for ik = 1:numel(partner)
        jk = partner(ik);
        if jk == 0 || jk == ik || ik > jk
            continue
        end
        vRows = 1:opts.outputNv;
        cRows = 1:opts.outputNc;
        C_data = align_selected_states_in_exact_clusters( ...
            C_data, E_data, selection.V_idx(vRows, ik), selection.V_idx(vRows, jk), ik, jk, opts);
        C_data = align_selected_states_in_exact_clusters( ...
            C_data, E_data, selection.C_idx(cRows, ik), selection.C_idx(cRows, jk), ik, jk, opts);
        C_data = align_family(C_data, E_data, selection.V_idx(vRows, ik), selection.V_idx(vRows, jk), ik, jk, opts);
        C_data = align_family(C_data, E_data, selection.C_idx(cRows, ik), selection.C_idx(cRows, jk), ik, jk, opts);
    end
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: ALIGN_SELECTED_STATES_IN_EXACT_CLUSTERS
%
% PURPOSE:
%   Internal helper used by TB_for_BSE and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function C_data = align_selected_states_in_exact_clusters( ...
        C_data, E_data, idxLeft, idxRight, ik, jk, opts)
    % If the target state lies in an exactly degenerate cluster, rotate only
    % inside that cluster to align it with the time-reversed source state.
    for ib = 1:numel(idxLeft)
        target = theta_spin_block(C_data(:, idxLeft(ib), ik));
        targetBand = idxRight(ib);
        [first, last] = exact_cluster_bounds(E_data(:, jk), targetBand, opts.exact_rotation_gap_eV);
        bands = first:last;
        if numel(bands) == 1
            continue
        end
        U = C_data(:, bands, jk);
        coefficients = U' * target;
        projection = U * coefficients;
        if norm(projection - target) > 1.0e-7
            continue
        end
        coefficients = coefficients / norm(coefficients);
        localTarget = find(bands == targetBand, 1);
        other = setdiff(1:numel(bands), localTarget, 'stable');
        rotation = zeros(numel(bands));
        rotation(:, localTarget) = coefficients;
        rotation(:, other) = null(coefficients');
        C_data(:, bands, jk) = U * rotation;
    end
end

function [first, last] = exact_cluster_bounds(E, band, tolerance)
    % Expand from one band index to the full contiguous exactly-degenerate
    % energy cluster around it.
    first = band;
    last = band;
    while first > 1 && abs(E(first) - E(first - 1)) <= tolerance
        first = first - 1;
    end
    while last < numel(E) && abs(E(last + 1) - E(last)) <= tolerance
        last = last + 1;
    end
end

function C_data = align_family(C_data, E_data, idxLeft, idxRight, ik, jk, opts)
    % Apply phase alignment for isolated labels and unitary Procrustes
    % alignment for selected degenerate label groups.
    NB = size(C_data, 1);
    if mod(NB, 2) ~= 0
        error('WannierBSE:TBTRGaugeBasis', 'TR gauge alignment requires equal spin blocks.');
    end
    first = 1;
    while first <= numel(idxLeft)
        last = first;
        while last < numel(idxLeft) && ...
                abs(E_data(idxLeft(last + 1), ik) - E_data(idxLeft(last), ik)) <= opts.exact_rotation_gap_eV && ...
                abs(E_data(idxRight(last + 1), jk) - E_data(idxRight(last), jk)) <= opts.exact_rotation_gap_eV
            last = last + 1;
        end
        local = first:last;
        target = theta_spin_block(C_data(:, idxLeft(local), ik));
        before = C_data(:, idxRight(local), jk);
        if numel(local) == 1
            overlap = before' * target;
            rotation = 1;
            if abs(overlap) > 0, rotation = overlap / abs(overlap); end
        else
            [left, ~, right] = svd(before' * target, 'econ');
            rotation = left * right';
        end
        after = before * rotation;
        C_data(:, idxRight(local), jk) = after;
        first = last + 1;
    end
end

function output = theta_spin_block(input)
    % Apply the spin-block representation of the time-reversal operator to
    % one or more coefficient vectors.
    half = size(input, 1) / 2;
    output = [conj(input(half + 1:end, :)); -conj(input(1:half, :))];
end

function diagnostics = time_reversal_diagnostics(E_data, selection, opts)
    % Summarize output-window TR energy and spin consistency without writing
    % diagnostic files to the package tree.
    partner = selection.TR_partner(:);
    nonself = find(partner > 0 & partner ~= (1:numel(partner)).');
    nV = min(opts.outputNv, size(selection.V_idx, 1));
    nC = min(opts.outputNc, size(selection.C_idx, 1));
    Venergy = nan(nV, numel(partner)); Vspin = nan(nV, numel(partner));
    Cenergy = nan(nC, numel(partner)); Cspin = nan(nC, numel(partner));
    for ik = 1:numel(partner)
        jk = partner(ik);
        if jk == 0, continue; end
        for ib = 1:nV
            Venergy(ib, ik) = abs(E_data(selection.V_idx(ib, ik), ik) - E_data(selection.V_idx(ib, jk), jk));
            if abs(selection.V_sz(ib, ik)) >= opts.spin_threshold && abs(selection.V_sz(ib, jk)) >= opts.spin_threshold
                Vspin(ib, ik) = abs(selection.V_sz(ib, ik) + selection.V_sz(ib, jk));
            end
        end
        for ib = 1:nC
            Cenergy(ib, ik) = abs(E_data(selection.C_idx(ib, ik), ik) - E_data(selection.C_idx(ib, jk), jk));
            if abs(selection.C_sz(ib, ik)) >= opts.spin_threshold && abs(selection.C_sz(ib, jk)) >= opts.spin_threshold
                Cspin(ib, ik) = abs(selection.C_sz(ib, ik) + selection.C_sz(ib, jk));
            end
        end
    end
    if isempty(nonself)
        maxEnergy = Inf; maxSpin = Inf; passed = false; reason = 'no_nonself_time_reversal_pairs';
        failure = struct('family', 'none', 'band', 0, 'k_index', 0);
    else
        energyErrors = [Venergy(:, nonself); Cenergy(:, nonself)];
        maxEnergy = max(energyErrors, [], 'all', 'omitnan');
        spinErrors = [Vspin(:, nonself); Cspin(:, nonself)];
        if all(isnan(spinErrors), 'all'), maxSpin = 0; else, maxSpin = max(spinErrors, [], 'all', 'omitnan'); end
        failure = locate_max_tr_spin_error(Vspin, Cspin, nonself);
        passed = maxEnergy <= opts.tr_energy_tolerance_eV;
        reason = '';
        if maxEnergy > opts.tr_energy_tolerance_eV, reason = 'time_reversal_energy'; end
    end
    diagnostics = struct('partner', partner, 'V_energy_error', Venergy, 'C_energy_error', Cenergy, ...
        'V_spin_reversal_error', Vspin, 'C_spin_reversal_error', Cspin, ...
        'max_energy_error', maxEnergy, 'max_spin_error', maxSpin, 'passed', passed, ...
        'failure_reason', reason, 'failure_location', failure);
end

function failure = locate_max_tr_spin_error(Vspin, Cspin, nonself)
    % Keep the largest spin-label mismatch location available for internal
    % error reporting and future debugging.
    vBlock = Vspin(:, nonself);
    cBlock = Cspin(:, nonself);
    if isempty(vBlock) || all(isnan(vBlock), 'all')
        vMax = -Inf;
    else
        vMax = max(vBlock, [], 'all', 'omitnan');
    end
    if isempty(cBlock) || all(isnan(cBlock), 'all')
        cMax = -Inf;
    else
        cMax = max(cBlock, [], 'all', 'omitnan');
    end
    if vMax >= cMax
        [row, col] = find(vBlock == vMax, 1);
        failure = struct('family', 'V', 'band', row, 'k_index', nonself(col));
    else
        [row, col] = find(cBlock == cMax, 1);
        failure = struct('family', 'C', 'band', row, 'k_index', nonself(col));
    end
end

function selection = refresh_selected_spins(C_data, selection, spin_blocks)
    % Recompute spin expectation values after any allowed degenerate-space
    % rotation or TR gauge alignment.
    for ik = 1:size(selection.V_idx, 2)
        for fullBand = 1:size(C_data, 2)
            selection.full_spectrum_sz(fullBand, ik) = ...
                spin_expectation(C_data(:, fullBand, ik), spin_blocks);
        end
        for ib = 1:size(selection.V_idx, 1)
            selection.V_sz(ib, ik) = spin_expectation(C_data(:, selection.V_idx(ib, ik), ik), spin_blocks);
        end
        for ib = 1:size(selection.C_idx, 1)
            selection.C_sz(ib, ik) = spin_expectation(C_data(:, selection.C_idx(ib, ik), ik), spin_blocks);
        end
    end
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: TB_DISPLAY_PATH
%
% PURPOSE:
%   Internal helper used by TB_for_BSE and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function displayPath = tb_display_path(pathValue)
anchors = {'Precomputed_data', 'User_input', 'Parameters', 'Exciton_data'};
displayPath = char(pathValue);
normalizedPath = strrep(displayPath, '\', '/');
for iAnchor = 1:numel(anchors)
    anchor = anchors{iAnchor};
    matchIndex = strfind(normalizedPath, anchor);
    if ~isempty(matchIndex)
        displayPath = normalizedPath(matchIndex(1):end);
        return
    end
end
displayPath = normalizedPath;
end
