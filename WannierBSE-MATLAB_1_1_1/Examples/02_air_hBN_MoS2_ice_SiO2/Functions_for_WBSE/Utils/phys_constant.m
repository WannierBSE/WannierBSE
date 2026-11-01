function [e,eps0] = phys_constant
%==========================================================================
% PHYS_CONSTANT
%
% DESCRIPTION:
%   Loads physical constants and converts charge and permittivity to the WBSE eV-Angstrom unit convention.
%
% INPUT ARGUMENTS:
%   None. Constants are loaded from physical_constant.mat.
%
% OUTPUT:
%   e : normalized elementary charge.
%   eps0 : vacuum permittivity in e^2/(eV Angstrom).
%==========================================================================

load('physical_constant.mat');
e_macro = e;
eps0_macro = eps0;

% change unit
clear e eps0 hbar m0
e = e_macro./e_macro;
eps0 = eps0_macro*(1/(e_macro*1e10));  % (e^2)/(eV*angstrom)
end
