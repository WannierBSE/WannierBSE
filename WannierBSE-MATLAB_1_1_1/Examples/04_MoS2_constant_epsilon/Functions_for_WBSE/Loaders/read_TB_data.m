function [En,Cn] = read_TB_data(Nk_all,nB_data,e,NB)
%==========================================================================
% READ_TB_DATA
%
% DESCRIPTION:
%   Parses one TB band data block into energies and complex Wannier-basis coefficients.
%
% INPUT ARGUMENTS:
%   Nk_all : number of k-point blocks.
%   nB_data : numeric content of one TB band file.
%   e : energy conversion factor.
%   NB : number of Wannier coefficients per k-point.
%
% OUTPUT:
%   En : band energy for each k-point.
%   Cn : k-point-by-basis complex coefficient matrix.
%==========================================================================



En = zeros(Nk_all,1);
Cn = zeros(NB,Nk_all);
for i = 1:Nk_all
    En(i) = nB_data(2+(i-1)*(NB+2),1).*e;
    Cn(1:NB,i) = nB_data((3+(i-1)*(NB+2)):((3+NB-1)+(i-1)*(NB+2)),1)...
                +1i*nB_data((3+(i-1)*(NB+2)):((3+NB-1)+(i-1)*(NB+2)),2);
end
Cn = transpose(Cn);


end
