function Vx_group = exchange_coulomb_source_group(chi_p, R_mNN, k, N_kpt, Cv, Cc, vv, cc, targets, j)
%==========================================================================
% EXCHANGE_COULOMB_SOURCE_GROUP
%
% DESCRIPTION:
%   Evaluates the short-range exchange blocks for one source band pair.
%   The source-dependent S2 intermediate is reused for every target block.
%
% INPUT ARGUMENTS:
%   chi_p, R_mNN : exchange pair-density tensor and retained translations.
%   k, N_kpt : k-point coordinates and number of k-points.
%   Cv, Cc : valence and conduction tight-binding coefficients.
%   vv, cc : source valence and conduction band indices.
%   targets : rows [pair_index, v, c] for target band pairs.
%   j : source k-point index.
%
% OUTPUT:
%   Vx_group : exchange-interaction blocks for the requested targets.
%==========================================================================

nr = size(R_mNN, 1);
nt = size(targets, 1);
S2_cache = cell(nr, nr);

for R_ind1 = 1:nr
    for R_ind2 = 1:nr
        chi_R1R2 = chi_p(:, :, :, :, R_ind1, R_ind2);
        S1 = sum((transpose(conj(Cv(j, :, vv))) .* chi_R1R2), 1);
        Cc_RR = transpose(Cc(j, :, cc) .* exp(1i * sum(k(j, :) .* R_mNN(R_ind2, 1:2), 2)));
        S2 = sum(Cc_RR .* (squeeze(S1)), 1);
        S2_cache{R_ind1, R_ind2} = squeeze(S2);
    end
end

Vx_group = complex(zeros(N_kpt, nt), zeros(N_kpt, nt));
for ti = 1:nt
    v = targets(ti, 2);
    c = targets(ti, 3);
    Vx = zeros(N_kpt, 1);
    for R_ind1 = 1:nr
        Cc_R = Cc(:, :, c) .* exp(1i * sum(k .* R_mNN(R_ind1, 1:2), 2));
        for R_ind2 = 1:nr
            Vx = Vx + sum((Cv(:, :, v) * S2_cache{R_ind1, R_ind2}) .* conj(Cc_R), 2);
        end
    end
    Vx_group(:, ti) = Vx;
end
end
