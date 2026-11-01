function [Q_out_SI, Etilde_out] = dielectric_function(control_file, phy, output_file_path)
%==========================================================================
% DIELECTRIC_FUNCTION
%
% DESCRIPTION:
%   Solves the layered quasi-2D electrostatic screening model and writes the effective in-plane
%   dielectric function epsilon(q).
%
% INPUT ARGUMENTS:
%   control_file : text file defining layer boundaries, layer counts, dielectric constants, and TMD
%   screening parameters.
%   phy : physical-constant structure loaded by the caller.
%   output_file_path : destination path for the generated epsilon table.
%
% OUTPUT:
%   Q_out_SI : q-grid written in SI-compatible inverse-length units used by the WBSE loaders.
%   Etilde_out : effective screened dielectric response sampled on the q-grid.
%==========================================================================

    params = parse_control_file(control_file);
    %------------------------------------------------------
    % 2. Validate layer architecture and dielectrics
    %------------------------------------------------------
    s = params.t_layers;
    t = params.s_layers;
    if ~isfield(params,'t_dielectric') || ~isfield(params,'s_dielectric')
        error('Control file must define t_dielectric and s_dielectric.');
    end
    if length(params.t_dielectric) ~= s
        error('Length of t_dielectric (%d) must match t_layers (%d).', ...
              length(params.t_dielectric), s);
    end
    if length(params.s_dielectric) ~= t
        error('Length of s_dielectric (%d) must match s_layers (%d).', ...
              length(params.s_dielectric), t);
    end
    E = [params.t_dielectric(:).' , NaN , params.s_dielectric(:).'];
    %------------------------------------------------------
    % Simulation grid
    %------------------------------------------------------
    q_max_input = 4;       % A^-1
    num_q       = 5000;
    num_z       = 100;
    %------------------------------------------------------
    % 3. Unit conversion: SI -> legacy
    %------------------------------------------------------
    e_SI    = phy.e;
    m0_SI   = phy.m0;
    hbar_SI = phy.hbar;
    Ang_to_nm = 0.1;
    conv_factor_E = (e_SI / m0_SI) * 1e-6;
    Z_legacy     = params.layer_boundaries * Ang_to_nm;
    q_max_legacy = q_max_input / Ang_to_nm;
    Epl_legacy = params.Epl * conv_factor_E;
    alpha      = params.alpha;
    ep_p       = params.ep_p;
    %------------------------------------------------------
    % 4. Legacy constants
    %------------------------------------------------------
    leg_epsilon0 = 3.1426e-7;
    leg_e         = 1;
    leg_m0        = 1;
    leg_hbar     = 1.1582e2;
    %------------------------------------------------------
    % 5. Pre-computation
    %------------------------------------------------------
    n = s + 1 + t;
    term1 = (leg_e^2 * leg_m0) / (pi^2 * leg_epsilon0 * leg_hbar^2);
    term2 = (3 * pi^2 * leg_epsilon0 * leg_m0 * Epl_legacy^2) / ...
            (leg_e^2 * leg_hbar^2);
    qTF = sqrt(term1 * term2^(1/3));
    h = Z_legacy(s) - Z_legacy(s+1);
    Q_vec_legacy = zeros(num_q,1);
    Etilde_out   = zeros(num_q,1);
    idx_vec = (1:num_z).';
    dz = h / num_z;
    z_grid  = -h/2 + idx_vec * dz;
    zp_grid = z_grid.';
    %------------------------------------------------------
    % 6. Main solver loop
    %------------------------------------------------------
    for i = 1:num_q
        q = i * (q_max_legacy / num_q);
        Q_vec_legacy(i) = q;
        term_TMD_1 = 1 / (ep_p - 1);
        term_TMD_2 = alpha * q^2 / qTF^2;
        term_TMD_3 = (leg_hbar^2 * q^2 / (2 * leg_m0 * Epl_legacy))^2;
        TMD_val = 1 + (term_TMD_1 + term_TMD_2 + term_TMD_3)^(-1);
        E_local = E;
        E_local(s+1) = TMD_val;
        A = zeros(2*n - 2);
        A(1,1) = exp(-q * Z_legacy(1)) / E_local(1);
        A(1,2) = -exp(-q * Z_legacy(1)) / E_local(2);
        A(1,3) = -exp(q * Z_legacy(1))  / E_local(2);
        A(2,1) = -exp(-q * Z_legacy(1));
        A(2,2) =  exp(-q * Z_legacy(1));
        A(2,3) = -exp(q * Z_legacy(1));
        if s > 1
            for j = 1:(s-1)
                row1 = 2*j + 1;
                row2 = 2*j + 2;
                A(row1,2*j)     =  exp(-q * Z_legacy(j+1)) / E_local(j+1);
                A(row1,2*j+1)   =  exp(q  * Z_legacy(j+1)) / E_local(j+1);
                A(row1,2*j+2)   = -exp(-q * Z_legacy(j+1)) / E_local(j+2);
                A(row1,2*j+3)   = -exp(q  * Z_legacy(j+1)) / E_local(j+2);
                A(row2,2*j)     = -exp(-q * Z_legacy(j+1));
                A(row2,2*j+1)   =  exp(q  * Z_legacy(j+1));
                A(row2,2*j+2)   =  exp(-q * Z_legacy(j+1));
                A(row2,2*j+3)   = -exp(q  * Z_legacy(j+1));
            end
        end
        if t > 1
            for k = 1:(t-1)
                row = 2*s + 2*k - 1;
                A(row,  2*s+2*k-2) = -exp(-q * Z_legacy(s+k)) / E_local(s+k);
                A(row,  2*s+2*k-1) = -exp(q  * Z_legacy(s+k)) / E_local(s+k);
                A(row,  2*s+2*k)   =  exp(-q * Z_legacy(s+k)) / E_local(s+k+1);
                A(row,  2*s+2*k+1) =  exp(q  * Z_legacy(s+k)) / E_local(s+k+1);
                A(row+1,2*s+2*k-2) =  exp(-q * Z_legacy(s+k));
                A(row+1,2*s+2*k-1) = -exp(q  * Z_legacy(s+k));
                A(row+1,2*s+2*k)   = -exp(-q * Z_legacy(s+k));
                A(row+1,2*s+2*k+1) =  exp(q  * Z_legacy(s+k));
            end
        end
        A(2*n-3,2*n-4) = -exp(-q * Z_legacy(n-1)) / E_local(n-1);
        A(2*n-3,2*n-3) = -exp(q  * Z_legacy(n-1)) / E_local(n-1);
        A(2*n-3,2*n-2) =  exp(q  * Z_legacy(n-1)) / E_local(n);
        A(2*n-2,2*n-4) =  exp(-q * Z_legacy(n-1));
        A(2*n-2,2*n-3) = -exp(q  * Z_legacy(n-1));
        A(2*n-2,2*n-2) =  exp(q  * Z_legacy(n-1));
        B_all = zeros(2*n - 2, num_z);
        B_all(2*s-1,:) =  exp(-q * (Z_legacy(s)    - zp_grid)) / TMD_val;
        B_all(2*s,:)   = -exp(-q * (Z_legacy(s)    - zp_grid));
        B_all(2*s+1,:) =  exp(-q * (zp_grid - Z_legacy(s+1))) / TMD_val;
        B_all(2*s+2,:) =  exp(-q * (zp_grid - Z_legacy(s+1)));
        scale_factors = max(abs(A), [], 2);
        scale_factors(scale_factors == 0) = 1;
        ws = warning('off', 'MATLAB:nearlySingularMatrix');
        warning('off', 'MATLAB:singularMatrix');
        warning('off', 'MATLAB:illConditionedMatrix');
        X_all = (A ./ scale_factors) \ (B_all ./ scale_factors);
        warning(ws);
        a0_vec = X_all(2*s,:);
        b0_vec = X_all(2*s+1,:);
        dist_matrix = abs(z_grid - zp_grid);
        v_pref = leg_e^2 / (2 * leg_epsilon0 * q);
        w_pref = leg_e^2 / (2 * TMD_val * leg_epsilon0 * q);
        vtilde = sum(exp(-q * dist_matrix),'all') * v_pref * dz^2;
        wtilde = sum((exp(-q*z_grid)*a0_vec + exp(q*z_grid)*b0_vec + ...
                      exp(-q*dist_matrix)),'all') * w_pref * dz^2;
        Etilde_out(i) = (vtilde / h^2) / (wtilde / h^2);
    end
    %------------------------------------------------------
    % 7. Output
    %------------------------------------------------------
    Q_out_SI = Q_vec_legacy * Ang_to_nm;
    Q_final  = [0.0; Q_out_SI];
    E_final  = [(E_local(1) + E_local(n)) / 2; Etilde_out];
    output_data = [Q_final, E_final];
    output_dir = fileparts(output_file_path);
    if ~exist(output_dir,'dir')
        mkdir(output_dir);
    end
    writematrix(output_data, output_file_path, 'Delimiter','tab');
end

function params = parse_control_file(filename)
%==========================================================================
% PARSE_CONTROL_FILE
%
% DESCRIPTION:
%   Parses numeric key-value parameters from the dielectric control file into a MATLAB structure.
%
% OUTPUT:
%   params : structure containing parsed parameter fields.
%==========================================================================

    fid = fopen(filename,'r');
    if fid == -1
        error('Cannot open control file: %s', filename);
    end
    params = struct();
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            parts = strsplit(line,'%');
            clean_line = strtrim(parts{1});
            if ~isempty(clean_line)
                kv = strsplit(clean_line,'=');
                if numel(kv) == 2
                    params.(strtrim(kv{1})) = str2num(strtrim(kv{2})); %#ok<ST2NM>
                end
            end
        end
    end
    fclose(fid);
end
