function output_log(DataPath, log_info)
%==========================================================================
% OUTPUT_LOG
%
% DESCRIPTION:
%   Writes a compact traceability report for the completed WannierBSE run.
%
% INPUT ARGUMENTS:
%   DataPath : output directory.
%   log_info : structure containing BSE basis, exchange, CPU, and timing metadata.
%
% OUTPUT:
%   None. The report is written to DataPath/WBSE_log.txt.
%==========================================================================

if ~exist(DataPath, 'dir')
    mkdir(DataPath);
end

log_file = fullfile(DataPath, 'WBSE_log.txt');
fid = fopen(log_file, 'w');
if fid < 0
    error('WBSE:OutputLogOpenFailed', 'Cannot open output log file: %s', log_file);
end

cleanup_obj = onCleanup(@() fclose(fid));

fprintf(fid, 'WannierBSE calculation report\n');
fprintf(fid, 'Generated at: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

write_section(fid, 'INPUT DATA PROVENANCE');
write_simple_value(fid, 'k-mesh source', log_info.kmesh_source);
write_simple_value(fid, 'TB data source', log_info.tb_source);
write_simple_value(fid, 'dielectric screening source', log_info.dielectric_source);
if log_info.Exchange_Interaction
    write_simple_value(fid, 'Wannier-function source', log_info.wannier_source);
end
fprintf(fid, '\n');

write_section(fid, 'BSE BASIS');
write_simple_value(fid, 'Number of k-points', sprintf('Nk = %d', log_info.N_kpt));
write_simple_value(fid, 'Valence bands used', sprintf('Nv = %d', log_info.Nv));
write_simple_value(fid, 'Conduction bands used', sprintf('Nc = %d', log_info.Nc));
fprintf(fid, '\n');

write_section(fid, 'COULOMB INTERACTION');
if log_info.Exchange_Interaction
    write_simple_value(fid, 'Short-range exchange interaction', 'enabled');
    if log_info.Use_Precomputed_Chi
        write_simple_value(fid, 'Exchange tensor source', 'loaded from precomputed data');
    else
        write_simple_value(fid, 'Exchange tensor source', 'computed during current run');
    end
    fprintf(fid, '\n');
    write_named_value(fid, 'Nearest-neighbor Wannier shell', 'M_max', log_info.M_max);
    write_named_value(fid, 'Largest short-range G-vector cutoff', 'G_max', log_info.G_max);
    write_named_value(fid, 'Wannier truncation probability cutoff', 'tol_probability', log_info.tol_probability);
    write_named_value(fid, 'M-neighbor distance tolerance', 'M_max_tolerance', log_info.M_max_tolerance);
    write_named_value(fid, 'Nearest-neighbor distance tolerance', 'nn_distance_tolerance', log_info.nn_distance_tolerance);
else
    write_simple_value(fid, 'Short-range exchange interaction', 'disabled');
end
fprintf(fid, '\n');

write_section(fid, 'CPU USAGE');
write_named_value(fid, 'TB calculations', 'Npar_TB', log_info.Npar_TB);
if log_info.Exchange_Interaction && ~log_info.Use_Precomputed_Chi
    write_named_value(fid, 'Exchange tensor calculation', 'Npar_Xi', log_info.Npar_Xi);
end
write_named_value(fid, 'BSE Hamiltonian construction', 'Npar_HBSE', log_info.Npar_HBSE);
fprintf(fid, '\n');

write_section(fid, 'COMPUTATION TIME');
if log_info.Exchange_Interaction && ~log_info.Use_Precomputed_Chi
    write_time_value(fid, 'rho overlap-density calculation', log_info.Time_rho);
    write_time_value(fid, 'short-range exchange tensor calculation', log_info.Time_exchange_tensor);
end
write_time_value(fid, 'BSE Hamiltonian matrix construction', log_info.Time_HBSE);
write_time_value(fid, 'BSE Hamiltonian diagonalization', log_info.Time_eigs);
write_time_value(fid, 'Total BSE solver stage', log_info.Time_BSE_total);
write_time_value(fid, 'Total WBSE execution', log_info.Time_WBSE_total);

end

function write_section(fid, title)
%==========================================================================
% WRITE_SECTION
%
% DESCRIPTION:
%   Writes a formatted section divider to the run log.
%==========================================================================

fprintf(fid, '==================== %s ====================\n', title);
end

function write_simple_value(fid, label, value_text)
%==========================================================================
% WRITE_SIMPLE_VALUE
%
% DESCRIPTION:
%   Writes a label-value line to the run log.
%==========================================================================

fprintf(fid, '%-45s = %s\n', label, value_text);
end

function write_named_value(fid, label, name, value)
%==========================================================================
% WRITE_NAMED_VALUE
%
% DESCRIPTION:
%   Writes a label, parameter name, and numeric value to the run log.
%==========================================================================

write_named_text(fid, label, name, format_scalar(value));
end

function write_named_text(fid, label, name, value_text)
%==========================================================================
% WRITE_NAMED_TEXT
%
% DESCRIPTION:
%   Writes a label, parameter name, and text value to the run log.
%==========================================================================

fprintf(fid, '%-45s %-22s = %s\n', label, name, value_text);
end

function write_time_value(fid, label, value)
%==========================================================================
% WRITE_TIME_VALUE
%
% DESCRIPTION:
%   Writes one timing entry in seconds to the run log.
%==========================================================================

fprintf(fid, '%-45s = %12.3f sec\n', label, value);
end

function txt = format_scalar(value)
%==========================================================================
% FORMAT_SCALAR
%
% DESCRIPTION:
%   Formats scalar numeric or text values for run-log output.
%
% OUTPUT:
%   txt : printable value string.
%==========================================================================

if isnumeric(value) && isscalar(value)
    txt = sprintf('%.12g', value);
else
    txt = char(string(value));
end
end
