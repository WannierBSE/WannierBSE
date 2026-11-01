function [q_mag, MinNdkG] = wave_vector_folding(q, b1, b2, G_dir_max)
%==========================================================================
% WAVE_VECTOR_FOLDING
%
% DESCRIPTION:
%   Folds momentum-transfer vectors by selecting the reciprocal-lattice shift with minimum magnitude.
%
% INPUT ARGUMENTS:
%   q : momentum-transfer vectors.
%   b1, b2 : reciprocal lattice vectors.
%
% OUTPUT:
%   q_mag : folded momentum magnitudes.
%   MinNdkG : index of the selected reciprocal shift.
%==========================================================================

if nargin < 4, G_dir_max = []; end
[Gi] = folding_G_vector(b1, b2, G_dir_max);

% --- Folding Logic ---
% Calculate q + G for all candidate G-vectors
qG = zeros(size(q, 1), 2, size(Gi, 1));
for gi = 1:size(Gi, 1)
    qG(:, :, gi) = q + Gi(gi, :);
end

% Compute magnitudes and find the minimum (closest to Gamma)
NqG = squeeze(sqrt(sum(qG.^2, 2)));
[q_mag, MinNdkG] = min(NqG, [], 2);

% Resolve only tied minima with an inversion-symmetric deterministic rule.
% Unique minima retain MATLAB's original first-minimum result exactly.
tie_tol_factor = 1e-12;
for qi = 1:size(q, 1)
    tied = find(abs(NqG(qi, :) - q_mag(qi)) <= ...
                tie_tol_factor * max(1, q_mag(qi)));
    if numel(tied) > 1
        [~, dominant_component] = max(abs(q(qi, :)));
        orientation = sign(q(qi, dominant_component));
        if orientation == 0
            orientation = 1;
        end
        transformed_G = orientation .* Gi(tied, :);
        selected_local = 1;
        for ti = 2:size(transformed_G, 1)
            if transformed_G(ti, 1) > transformed_G(selected_local, 1) || ...
                    (transformed_G(ti, 1) == transformed_G(selected_local, 1) && ...
                     transformed_G(ti, 2) > transformed_G(selected_local, 2))
                selected_local = ti;
            end
        end
        MinNdkG(qi) = tied(selected_local);
    end
end

end
