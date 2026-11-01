function [vc_band_BSE] = generate_vc_band(Nv, Nc)
%==========================================================================
% GENERATE_VC_BAND
%
% DESCRIPTION:
%   Builds the ordered valence-conduction band-pair table used to enumerate the BSE basis at each k-point.
%
% INPUT ARGUMENTS:
%   Nv : number of valence bands.
%   Nc : number of conduction bands.
%
% OUTPUT:
%   vc_band_BSE : (Nc*Nv)-by-2 table with rows [v, c].
%==========================================================================

vc_band_BSE = zeros(Nc * Nv, 2);

% --- Index Generation Loop ---
% Iterate through valence and conduction bands to form all possible pairs
for vi = 1:Nv
    for ci = 1:Nc
        % Linear index mapping: (ci + (vi-1)*Nc)
        vc_band_BSE(ci + (vi-1) * Nc, :) = [vi, ci];
    end
end

end
