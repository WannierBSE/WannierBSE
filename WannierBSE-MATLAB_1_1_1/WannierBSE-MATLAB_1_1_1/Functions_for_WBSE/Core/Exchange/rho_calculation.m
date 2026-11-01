function [rho] = rho_calculation(W_up,W_down,W_up_t,W_down_t,G_all,R_mNN,n1_n2_mNN,...
                                        zcut_min,zcut_max,Na1,Na2,Na3,shift_a1,shift_a2,shift_a3,tau,...
                                        N_atoms,d_max,Basis_tau_index,Mmax_tolerance,tin_all,Nz_t,Na1_t,Na2_t,...
                                        kex,a1,a2,u_point_t,v_point_t,u_t,hu,hv,J,kexTol)
%==========================================================================
% RHO_CALCULATION
%
% DESCRIPTION:
%   Computes Fourier-transformed Wannier pair densities needed for the
%   short-range exchange tensor. The validated operator path evaluates each
%   in-plane Wij through fixed, response-built separable factors. The scalar
%   spline/integration helpers remain available as the numerical reference.
%
% INPUT ARGUMENTS:
%   W_up, W_down and truncated/grid arguments : spinor Wannier data.
%   G_all, R_mNN, n1_n2_mNN : reciprocal vectors and retained translations.
%   Exchange geometry, spline, grid, and tolerance arguments : preprocessing data.
%
% OUTPUT:
%   rho : Fourier-transformed Wannier pair-density tensor.
%==========================================================================

Nuv = Na1_t*Na2_t;
t_factor = tic;
[U,V,factor_meta] = rho_operator_build_factors(G_all,kex,a1,a2,u_t(1:Nuv), ...
                                                u_point_t,v_point_t,kexTol,hu,hv,J);

rho = complex(zeros(size(W_up_t,2),size(W_up_t,2),length(zcut_min:zcut_max), ...
                    size(R_mNN,1),size(G_all,1)));
t_kernel = tic;
for Ri=1:size(R_mNN,1)
    R = R_mNN(Ri,:);

    %------------W(r-R)----------
    n1 = n1_n2_mNN(Ri,1);
    n2 = n1_n2_mNN(Ri,2);
    Wl_up_t = complex(zeros(size(W_up_t)));
    Wl_down_t = complex(zeros(size(W_up_t)));
    for wj=1:size(W_up,2)
        Reshape_W_up = reshape(W_up(:,wj),[Na1,Na2,Na3]);
        Reshape_W_up = circshift(Reshape_W_up,[n1*shift_a1,n2*shift_a2,shift_a3]);
        Wl_up = reshape(Reshape_W_up,[Na1*Na2*Na3,1]);
        Wl_up_t(:,wj) = Wl_up(tin_all);

        Reshape_W_down = reshape(W_down(:,wj),[Na1,Na2,Na3]);
        Reshape_W_down = circshift(Reshape_W_down,[n1*shift_a1,n2*shift_a2,shift_a3]);
        Wl_down = reshape(Reshape_W_down,[Na1*Na2*Na3,1]);
        Wl_down_t(:,wj) = Wl_down(tin_all);
    end
    %------------W(r-R)----------

    [j_index_tot] = find_j_index_in_rho_integral(R,tau,N_atoms,d_max,Basis_tau_index,Mmax_tolerance);
    for jj=1:size(W_up_t,2)
        W0i_up_t = W_up_t(:,jj);
        W0i_down_t = W_down_t(:,jj);
        j_index = j_index_tot{jj};
        for j=j_index
            WRj_up_t = Wl_up_t(:,j);
            WRj_down_t = Wl_down_t(:,j);
            Wij = (conj(W0i_up_t).*WRj_up_t)+(conj(W0i_down_t).*WRj_down_t);

            for zi=1:Nz_t
                ind_z = (1:Nuv)+(zi-1)*Nuv;
                Wij_z = reshape(Wij(ind_z),Na1_t,Na2_t);
                rho_temp_G = rho_operator_evaluate_plane(U,V,Wij_z);
                rho(jj,j,zi,Ri,:) = reshape(rho_temp_G,1,1,1,1,size(G_all,1));
            end
        end
    end
end
end
