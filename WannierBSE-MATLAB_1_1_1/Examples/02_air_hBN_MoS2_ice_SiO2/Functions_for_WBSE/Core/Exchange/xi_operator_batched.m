function Xi = xi_operator_batched(rho1,rho2,NB,b1,z_point_t,jj_index,...
    j_index_tot,ll_index,l_index_tot,w,M_small,M_large)
%==========================================================================
% XI_OPERATOR_BATCHED
%
% DESCRIPTION:
%   Evaluates and scatters one batched operator Xi block. The operators are
%   passed in preconstructed form: w has grid lifetime, while M_small and
%   M_large have G lifetime. Pair-list traversal follows xi_integral.
%
% INPUT ARGUMENTS:
%   rho1, rho2 : pair-density tensors.
%   NB, b1, z_point_t : basis size, reciprocal vector, and z grid.
%   jj_index, j_index_tot, ll_index, l_index_tot : pair neighborhoods.
%   w, M_small, M_large : preconstructed integration operators.
%
% OUTPUT:
%   Xi : batched Xi operator contribution.
%
% NOTES:
%   The non-conjugating transpose in the operator contraction is intentional.
%==========================================================================

if ndims(rho1) ~= 3 || ndims(rho2) ~= 3
    error('xi_operator_batched:InvalidRho','rho inputs must be NB-by-NB-by-Nz.');
end
Nz = size(rho1,3);
if ~isequal(size(rho2),[NB,NB,Nz])
    error('xi_operator_batched:InvalidRhoSize','rho1 and rho2 dimensions disagree.');
end
if ~isequal(size(w),[Nz,1]) || ~isequal(size(M_small),[Nz,Nz]) || ...
        ~isequal(size(M_large),[Nz,Nz])
    error('xi_operator_batched:InvalidOperatorSize','Operator dimensions disagree with rho.');
end

pairs1 = local_pairs(jj_index,j_index_tot);
pairs2 = local_pairs(ll_index,l_index_tot);
P1 = size(pairs1,1);
P2 = size(pairs2,1);
rho1_pair_signals = complex(zeros(Nz,P1),zeros(Nz,P1));
rho2_pair_signals = complex(zeros(Nz,P2),zeros(Nz,P2));
for p = 1:P1
    rho1_pair_signals(:,p) = reshape(rho1(pairs1(p,1),pairs1(p,2),:),Nz,1);
end
for p = 1:P2
    rho2_pair_signals(:,p) = reshape(rho2(pairs2(p,1),pairs2(p,2),:),Nz,1);
end

Q1 = conj(rho1_pair_signals);
B = M_small*rho2_pair_signals + M_large*rho2_pair_signals;
X = Q1.' * (w .* B);

Xi = complex(zeros(NB,NB,NB,NB),zeros(NB,NB,NB,NB));
for a = 1:P1
    jj = pairs1(a,1);
    j = pairs1(a,2);
    for b = 1:P2
        ll = pairs2(b,1);
        l = pairs2(b,2);
        Xi(jj,j,ll,l) = X(a,b);
    end
end
end

function pairs = local_pairs(index,tot)
pairs = zeros(0,2);
for jj = index
    range = tot{jj};
    pairs = [pairs; [repmat(jj,numel(range),1), range(:)]]; %#ok<AGROW>
end
end
