function WF_to_mat_parallel()
%==========================================================================
% WF_TO_MAT_PARALLEL
%
% DESCRIPTION:
%   Runs the production parallel fused Wannier preprocessing workflow and
%   writes a validated transactional WBSE Wannier cache.
%
% INPUT ARGUMENTS:
%   None. Project paths and preprocessing controls are loaded from the
%   standardized project configuration.
%
% OUTPUT:
%   None. The completed Wannier cache is written to the configured output path.
%==========================================================================
Paths = init_wannier_paths();
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
params = read_params(fullfile(Paths.parameters, 'exchange_control.txt'));
cleanup_xsf = false;
if isfield(params, 'cleanup_xsf')
    cleanup_xsf = parse_cleanup_xsf(params.cleanup_xsf);
end
final = Paths.final_output;
if isfolder(final)
    % A complete cache is protected from accidental overwrite.  An
    % incomplete cache, however, cannot be reused safely: remove the entire
    % directory so every Wannier output is regenerated consistently.
    [cache_ok, cache_reason] = validate_wf_cache(final);
    if cache_ok
        error('WFOptimized:OutputExists', ...
              'Refusing to overwrite existing complete WF cache: %s', final);
    end
    fprintf('[CACHE] Wannier: incomplete cache detected (%s). Clearing %s before rebuild...\n', ...
            cache_reason, final);
    [removed, remove_message] = rmdir(final, 's');
    if ~removed
        error('WFOptimized:CleanupFailed', ...
              'Unable to remove incomplete WF cache %s: %s', ...
              final, remove_message);
    end
end
stamp = datestr(now, 'yyyymmdd_HHMMSSFFF');
incomplete = [final, '.incomplete_', stamp];
mkdir(incomplete);
total = tic;

geometry_timer = tic;
g = wf_prepare_geometry_optimized(root);
shared_geometry_seconds = toc(geometry_timer);

nb = 22;
map = [1;12;2;13;3;14;4;15;5;16;6;17;7;18;8;19;9;20;10;21;11;22];
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', 4);
end
up_cell = cell(1, nb);
down_cell = cell(1, nb);
norm_before = zeros(1, nb);
norm_const = zeros(1, nb);
compute_seconds = zeros(1, nb);
parallel_timer = tic;
parfor source = 1:nb
    c = wf_process_one_optimized(g, source);
    up_cell{source} = c.W_up;
    down_cell{source} = c.W_down;
    norm_before(source) = c.norm_before;
    norm_const(source) = c.norm_const;
    compute_seconds(source) = c.timing.candidate_total_s;
end
parallel_seconds = toc(parallel_timer);

W_up = zeros(g.n, nb, 'like', 1i);
W_down = zeros(g.n, nb, 'like', 1i);
for source = 1:nb
    destination = map(source);
    W_up(:, destination) = up_cell{source};
    W_down(:, destination) = down_cell{source};
end
clear up_cell down_cell

save(fullfile(incomplete, 'WF_up.mat'), 'W_up', '-v7.3', '-nocompression');
save(fullfile(incomplete, 'WF_down.mat'), 'W_down', '-v7.3', '-nocompression');
r = g.r;
save(fullfile(incomplete, 'r.mat'), 'r', '-v7.3');
write_gridinfo(fullfile(incomplete, 'GridInfo.txt'), g);
write_atoms(fullfile(incomplete, 'WF_AtomPosition.txt'), g);
reorder_WF_centers(fullfile(Paths.parameters, 'WF_centers.txt'), ...
    fullfile(incomplete, 'WF_centers_WBSE.txt'));

fid = fopen(fullfile(incomplete, 'normalization_diagnostics.csv'), 'w');
fprintf(fid, 'source_wf,destination_column,norm_before,norm_const,norm_deviation\n');
for source = 1:nb
    fprintf(fid, '%d,%d,%.17g,%.17g,%.17g\n', source, map(source), ...
        norm_before(source), norm_const(source), ...
        abs(norm_before(source) - norm_const(source)^2));
end
fclose(fid);
fid = fopen(fullfile(incomplete, 'COMPLETE'), 'w');
fprintf(fid, 'Parallel WF cache complete.\n');
fclose(fid);

[ok, reason] = validate_wf_cache(incomplete);
if ~ok
    error('WFOptimized:Validation', 'Incomplete cache: %s', reason);
end
[ok, message] = movefile(incomplete, final);
if ~ok
    error('WFOptimized:Finalize', '%s', message);
end
fprintf('[DONE ] Wannier: parallel preprocessing completed in %.2f s. Cleanup flag: %d\n', ...
    toc(total), cleanup_xsf);
end

function write_gridinfo(path, g)
fid = fopen(path, 'w');
fprintf(fid, '%d %d %d \n', g.dims);
fprintf(fid, '%.14f %.14f %.14f \n', g.origin);
for i = 1:3
    fprintf(fid, '%.14f %.14f %.14f', g.vectors(i, :));
    if i < 3, fprintf(fid, ' \n'); end
end
fclose(fid);
end

function write_atoms(path, g)
fid = fopen(path, 'w');
fprintf(fid, '%.14f %.14f %.14f \n', g.origin);
for i = 1:numel(g.atoms)
    fprintf(fid, '%.14f %.14f %.14f \n', g.atoms{i}.position);
end
fclose(fid);
end
