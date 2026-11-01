function [HBSE, hbse_timing] = build_HBSE_flattened( ...
    Npar_HBSE, Ev, Ec, Cv, Cc, k, b1, b2, d, q_data, epsQ2D_data, Wq0, wi, tau, Basis_tau_index, ...
    WF_center, N_atoms, e, eps0, chi_p, R_mNN, Exchange_Interaction, Nv, Nc, G_dir_max)
%==========================================================================
% BUILD_HBSE_FLATTENED
%
% DESCRIPTION:
%   Builds the flattened Bethe-Salpeter Hamiltonian block-by-block using
%   the independent-particle term, screened direct interaction, and optional
%   short-range exchange interaction. Serial and parallel source-k-point
%   execution use the same numerical assembly path.
%
% INPUT ARGUMENTS:
%   Npar_HBSE : maximum number of workers for HBSE construction.
%   Ev, Ec, Cv, Cc : valence/conduction energies and TB coefficients.
%   k, b1, b2 : k-point coordinates and reciprocal lattice vectors.
%   d, q_data, epsQ2D_data : thickness and dielectric screening data.
%   Wq0, wi : singular-cell Coulomb values and k-space weights.
%   tau, Basis_tau_index, WF_center, N_atoms : Wannier geometry metadata.
%   e, eps0 : physical charge and vacuum permittivity constants.
%   chi_p, R_mNN : precomputed exchange tensor and retained translations.
%   Exchange_Interaction : logical switch enabling short-range exchange.
%   Nv, Nc : numbers of valence and conduction bands.
%   G_dir_max : direct reciprocal-vector cutoff; omitted or empty defaults to 1.
%
% OUTPUT:
%   HBSE : flattened BSE Hamiltonian matrix.
%   hbse_timing : optional timing structure for direct and exchange stages.
%
% NOTES:
%   The assembly follows H_BSE = K + weight .* (Vd + Vx), with Vx omitted
%   when Exchange_Interaction is false. Vd/Vx timings are accumulated inside
%   the HBSE pass.
%==========================================================================
if nargin < 25, G_dir_max = []; end

Nk = size(k, 1);
bands = generate_vc_band(Nv, Nc);
nPairs = size(bands, 1);
collect_timing = nargout > 1;

for subspace = 1:nPairs
    fprintf('  -> Processing subspace: v%d, c%d\n', bands(subspace, 1), bands(subspace, 2));
end

pairs = zeros(nPairs * (nPairs + 1) / 2, 2);
n = 0;
for base = 1:nPairs
    for target = base:nPairs
        n = n + 1;
        pairs(n, :) = [base, target];
    end
end

tasks = zeros(size(pairs, 1) * Nk, 2);
n = 0;
for p = 1:size(pairs, 1)
    for j = 1:Nk
        n = n + 1;
        tasks(n, :) = [p, j];
    end
end

columns = cell(size(tasks, 1), 1);
vd_worker_times = zeros(size(tasks, 1), 1);
vx_worker_times = zeros(size(tasks, 1), 1);
vd_phase_timer = tic;
[Cc_tilde, Cv_tilde] = phase_tau_in_Vd(k, Cc, Cv, tau, WF_center);
vd_phase_time = toc(vd_phase_timer);

source_groups = build_exchange_source_groups(pairs, bands);
grouped_columns = cell(Nk, 1);
vd_group_worker_times = zeros(Nk, 1);
vx_group_worker_times = zeros(Nk, 1);
% Explicit broadcast aliases for MATLAB R2021a process-worker analysis.
wi_local = wi;
Wq0_local = Wq0;

if Npar_HBSE <= 1
    for j = 1:Nk
        [grouped_columns{j}, vd_group_worker_times(j), vx_group_worker_times(j)] = build_grouped_columns_for_source(collect_timing, ...
            k, b1, b2, d, q_data, epsQ2D_data, Wq0_local(j), wi_local(j), tau, Basis_tau_index, N_atoms, ...
            Cc_tilde, Cv_tilde, pairs, bands, j, e, eps0, Ec, Ev, Exchange_Interaction, chi_p, R_mNN, Cv, Cc, source_groups, wi_local, G_dir_max);
    end
else
    parfor (j = 1:Nk, Npar_HBSE)
        [grouped_columns{j}, vd_group_worker_times(j), vx_group_worker_times(j)] = build_grouped_columns_for_source(collect_timing, ...
            k, b1, b2, d, q_data, epsQ2D_data, Wq0_local(j), wi_local(j), tau, Basis_tau_index, N_atoms, ...
            Cc_tilde, Cv_tilde, pairs, bands, j, e, eps0, Ec, Ev, Exchange_Interaction, chi_p, R_mNN, Cv, Cc, source_groups, wi_local, G_dir_max);
    end
end

if collect_timing
    hbse_timing = struct();
    hbse_timing.Time_Vd_phase_precompute = vd_phase_time;
    hbse_timing.Time_Vd_worker_accumulated = sum(vd_group_worker_times);
    hbse_timing.Time_Vx_worker_accumulated = sum(vx_group_worker_times);
end

% ========================================================================
% GLOBAL HBSE MATRIX ASSEMBLY
%
% The BSE Hamiltonian is assembled block-by-block rather than through one
% single matrix expression.  For each electron-hole transition, the
% effective matrix element is
%
%   H_BSE = K + weight .* (Vd + Vx),
%
% where K is the independent-particle electron-hole energy, Vd is the
% screened direct Coulomb interaction, Vx is the optional short-range
% exchange interaction, and
%
%   weight = sqrt(w_i * w_j) / (2*pi)^2.
%
% The assembly is represented in two interaction steps:
%
%   1) H_BSE <- K + weight .* Vd,     K = Ec - Ev
%   2) H_BSE <- H_BSE + weight .* Vx
%
% Therefore, elementwise,
%
%   H_BSE_ij = K_ij + sqrt(w_i*w_j)/(2*pi)^2 * (Vd_ij + Vx_ij).
%
% The direct term is always included.  The exchange term is included only
% when Exchange_Interaction is true.  The individual terms are assembled
% below in build_grouped_columns_for_source().
% ========================================================================
HBSE = complex(zeros(Nv * Nc * Nk), zeros(Nv * Nc * Nk));
for p = 1:size(pairs, 1)
    B = complex(zeros(Nk), zeros(Nk));
    for j = 1:Nk
        source_columns = grouped_columns{j};
        B(:, j) = source_columns(:, p);
    end

    base = pairs(p, 1);
    target = pairs(p, 2);
    vv = bands(base, 1);
    cc = bands(base, 2);
    v = bands(target, 1);
    c = bands(target, 2);

    idx1 = (1:Nk) + ((c - 1) + (v - 1) * Nc) * Nk;
    idx2 = (1:Nk) + ((cc - 1) + (vv - 1) * Nc) * Nk;
    HBSE(idx1, idx2) = B;

    if v ~= vv || c ~= cc
        % B is already symmetrically weighted; complete the reverse block directly.
        HBSE(idx2, idx1) = ctranspose(B);
    end
end
return
end

function [source_columns, vd_elapsed, vx_elapsed] = build_grouped_columns_for_source( ...
    collect_timing, k, b1, b2, d, q_data, epsQ2D_data, Wq0_j, wi_j, tau, Basis_tau_index, N_atoms, ...
    Cc_tilde, Cv_tilde, pairs, bands, j, e, eps0, Ec, Ev, Exchange_Interaction, chi_p, R_mNN, Cv, Cc, ...
    source_groups, wi_all, G_dir_max)
% ========================================================================
% PER-SOURCE-K-POINT HBSE COLUMN ASSEMBLY
% ========================================================================

% ------------------------------------------------------------------------
% 1. Direct screened Coulomb interaction Vd
% ------------------------------------------------------------------------
timer_value = 0;
if collect_timing
    timer_value = tic;
end
Vd_group = direct_coulomb_source_group(k, b1, b2, d, q_data, epsQ2D_data, Wq0_j, wi_j, tau, Basis_tau_index, N_atoms, Cc_tilde, Cv_tilde, pairs, bands, j, e, eps0, G_dir_max);
if collect_timing
    vd_elapsed = toc(timer_value);
else
    vd_elapsed = 0;
end

% ------------------------------------------------------------------------
% 2. Independent-particle electron-hole kinetic energy term K and integration weight
% ------------------------------------------------------------------------
Nk = size(k, 1);
source_columns = complex(zeros(Nk, size(pairs, 1)), zeros(Nk, size(pairs, 1)));
for p = 1:size(pairs, 1)
    base = pairs(p, 1);
    target = pairs(p, 2);
    vv = bands(base, 1);
    cc = bands(base, 2);
    v = bands(target, 1);
    c = bands(target, 2);

    KE = zeros(Nk, 1);
    KE(j) = (Ec(j, c) - Ev(j, v)) * (c == cc) * (v == vv);
    symmetric_weight = sqrt(wi_all(:) .* wi_j) / (2 * pi)^2;

    % Step 1 (direct part): start from K + weight .* Vd.
    source_columns(:, p) = KE + symmetric_weight .* Vd_group(:, p);
end

% ------------------------------------------------------------------------
% 3. Optional short-range exchange interaction Vx
% ------------------------------------------------------------------------
vx_timer = 0;
if collect_timing
    vx_timer = tic;
end
if Exchange_Interaction
    for gi = 1:numel(source_groups)
        group = source_groups(gi);
        Vx_group = exchange_coulomb_source_group(chi_p, R_mNN, k, Nk, Cv, Cc, group.vv, group.cc, group.targets, j);
        for ti = 1:size(group.targets, 1)
            p = group.targets(ti, 1);
            % Step 2 (exchange part): add weight .* Vx, giving
            % H_BSE = K + weight .* (Vd + Vx).
            source_columns(:, p) = source_columns(:, p) + symmetric_weight .* Vx_group(:, ti);
        end
    end
end
if collect_timing
    vx_elapsed = toc(vx_timer);
else
    vx_elapsed = 0;
end
end

function groups = build_exchange_source_groups(pairs, bands)
nSources = max(pairs(:, 1));
groups = repmat(struct('base', 0, 'vv', 0, 'cc', 0, 'targets', []), nSources, 1);
for base = 1:nSources
    ids = find(pairs(:, 1) == base);
    groups(base).base = base;
    groups(base).vv = bands(base, 1);
    groups(base).cc = bands(base, 2);
    targets = zeros(numel(ids), 3);
    for ii = 1:numel(ids)
        p = ids(ii);
        target = pairs(p, 2);
        targets(ii, :) = [p, bands(target, 1), bands(target, 2)];
    end
    groups(base).targets = targets;
end
end
