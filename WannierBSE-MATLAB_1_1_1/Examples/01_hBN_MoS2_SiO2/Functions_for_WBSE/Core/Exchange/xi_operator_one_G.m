function [XiG,stats] = xi_operator_one_G(rho,R_mNN,G_all,g,kex,tau,...
    N_atoms,d_max,Basis_tau_index,Mmax_tolerance,e,eps0,z_point_t,b1,...
    hf,hf0,w)
%==========================================================================
% XI_OPERATOR_ONE_G
%
% DESCRIPTION:
%   Evaluates one reciprocal-vector contribution to the operator-based Xi
%   exchange tensor. The output retains the legacy six-dimensional layout.
%
% INPUT ARGUMENTS:
%   rho, R_mNN, G_all, g, kex : pair densities, translations, reciprocal
%       vectors, selected G index, and exchange momentum.
%   tau, N_atoms, d_max, Basis_tau_index, Mmax_tolerance : geometry controls.
%   e, eps0, z_point_t, b1 : physical constants, z grid, and lattice vector.
%   hf, hf0, w : spline function handles and fixed-grid operator.
%
% OUTPUT:
%   XiG : one-G Xi tensor with legacy NB-by-NB-by-NB-by-NB-by-nR-by-nR layout.
%   stats : timing and operator-construction diagnostics.
%==========================================================================

NB = size(rho,1);
Nz = size(rho,3);
nR = size(R_mNN,1);
if size(rho,4) ~= nR || size(rho,5) ~= size(G_all,1)
    error('xi_operator_one_G:InputSize','rho, R_mNN, and G_all dimensions disagree.');
end
if ~isequal(size(w),[Nz,1])
    error('xi_operator_one_G:InvalidW','w must be Nz-by-1 and preconstructed.');
end
G = G_all(g,:);
jj_index = 1:NB;
ll_index = 1:NB;

tAll = tic;
tSetup = tic;
pairLists = cell(nR,1);
pairCounts = zeros(nR,1);
for r = 1:nR
    pairLists{r} = find_j_index_in_rho_integral(R_mNN(r,:),tau,N_atoms,...
        d_max,Basis_tau_index,Mmax_tolerance);
    pairCounts(r) = sum(cellfun(@numel,pairLists{r}));
end
tSetup = toc(tSetup);

tM = tic;
[M_small,M_large] = xi_operator_build_B(hf,hf0,z_point_t,G,kex,b1);
tOperator = toc(tM);

V_kex_G = ((e^2)/(2*eps0))*(1./sqrt(sum((kex+[G,0]).^2)));
XiG = complex(zeros(NB,NB,NB,NB,nR,nR), ...
    zeros(NB,NB,NB,NB,nR,nR));
tB = zeros(1,nR);
tContract = zeros(nR,nR);
tScatter = zeros(nR,nR);

for r2 = 1:nR
    pairs2 = local_pairs(ll_index,pairLists{r2});
    rho2_pair_signals = local_signals(rho(:,:,:,r2,g),pairs2,Nz);
    tb = tic;
    B = M_small*rho2_pair_signals + M_large*rho2_pair_signals;
    tB(r2) = toc(tb);

    for r1 = 1:nR
        pairs1 = local_pairs(jj_index,pairLists{r1});
        rho1_pair_signals = local_signals(rho(:,:,:,r1,g),pairs1,Nz);
        Q1 = conj(rho1_pair_signals);
        tx = tic;
        X = Q1.' * (w .* B);
        tContract(r1,r2) = toc(tx);

        ts = tic;
        for a = 1:size(pairs1,1)
            jj = pairs1(a,1);
            j = pairs1(a,2);
            for b = 1:size(pairs2,1)
                ll = pairs2(b,1);
                l = pairs2(b,2);
                XiG(jj,j,ll,l,r1,r2) = V_kex_G*X(a,b);
            end
        end
        tScatter(r1,r2) = toc(ts);
    end
end

stats.G = G;
stats.g_index = g;
stats.pair_counts = pairCounts;
stats.pair_lists = pairLists;
stats.w_reused = true;
stats.operator_build_count = 1;
stats.B_batch_count = nR;
stats.t_setup = tSetup;
stats.t_operator = tOperator;
stats.t_B = tB;
stats.t_contract = tContract;
stats.t_scatter = tScatter;
stats.t_total = toc(tAll);
stats.t_kernel_excluding_operator = stats.t_total - tOperator;
stats.memory_peak_bytes = local_memory_estimate(NB,nR,Nz,max(pairCounts));
stats.output_size = size(XiG);
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: LOCAL_PAIRS
%
% PURPOSE:
%   Internal helper used by xi_operator_one_G and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function pairs = local_pairs(index,tot)
pairs = zeros(0,2);
for jj = index
    range = tot{jj};
    pairs = [pairs; [repmat(jj,numel(range),1),range(:)]]; %#ok<AGROW>
end
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: LOCAL_SIGNALS
%
% PURPOSE:
%   Internal helper used by xi_operator_one_G and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function signals = local_signals(rhoBlock,pairs,Nz)
signals = complex(zeros(Nz,size(pairs,1)),zeros(Nz,size(pairs,1)));
for p = 1:size(pairs,1)
    signals(:,p) = reshape(rhoBlock(pairs(p,1),pairs(p,2),:),Nz,1);
end
end

%--------------------------------------------------------------------------
% INTERNAL HELPER: LOCAL_MEMORY_ESTIMATE
%
% PURPOSE:
%   Internal helper used by xi_operator_one_G and kept local to this module.
%   The function signature below defines its local data contract.
%--------------------------------------------------------------------------
function bytes = local_memory_estimate(NB,nR,Nz,maxP)
bytesPer = 16;
outputBytes = NB^4*nR^2*bytesPer;
signalBytes = 4*Nz*maxP*bytesPer;
xBytes = maxP^2*bytesPer;
operatorBytes = (2*Nz^2+Nz)*bytesPer;
bytes = outputBytes + signalBytes + xBytes + operatorBytes;
end
