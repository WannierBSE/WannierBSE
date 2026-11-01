function [G] = g_vector_v2(Glim,b1,b2)
%==========================================================================
% G_VECTOR_V2
%
% DESCRIPTION:
%   Generates in-plane reciprocal lattice vectors within the exchange cutoff range.
%
% INPUT ARGUMENTS:
%   Glim : reciprocal-index cutoff.
%   b1, b2 : reciprocal lattice vectors.
%
% OUTPUT:
%   G : list of reciprocal lattice vectors.
%==========================================================================


if (Glim==0)
G = [0,0];
else
Gi = -ceil(Glim):ceil(Glim);
Gj = -ceil(Glim):ceil(Glim);
for i=1:length(Gj)
    for j=1:length(Gi)
        G_component(i+(j-1)*length(Gi),1) = Gi(i);
        G_component(i+(j-1)*length(Gj),2) = Gj(j);
    end
end

for i=1:length(G_component)
    G_parallelogram(i,:) = G_component(i,1)*transpose(b1)+G_component(i,2)*transpose(b2);
    Gnorm(i,1) = sqrt(sum(G_parallelogram(i,:).^2));
end

Gmax = Glim*norm(b1);
G_hexagonal = find((sqrt(sum(G_parallelogram.^2,2)))<=(Gmax*1.005));
G = G_parallelogram(G_hexagonal,:);
end


end
