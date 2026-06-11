function [a, b] = getNASAcoeffs_air5(T)
%GETNASACOEFFS_AIR5
% Returns NASA-9 coefficients (a,b) for a given temperature T.
%
% Output species order:
%   [N, O, NO, N2, O2]
%
% Automatically selects:
%   200–1000 K   -> a1, b1
%   1000–6000 K  -> a2, b2
%   >6000 K      -> a3, b3
%
% Compatible with nasa9eval_G / nasa9eval_H

% Load database
[species_thermo_structs, ~, ~] = thermodynamicsModels();

% Order in thermodynamicsModels():
% {N, O, NO, N2, O2}
Nstruct  = species_thermo_structs{1};
Ostruct  = species_thermo_structs{2};
NOstruct = species_thermo_structs{3};
N2struct = species_thermo_structs{4};
O2struct = species_thermo_structs{5};

% Allocate
a = zeros(7,5);
b = zeros(2,5);

% Temperature selection
if T <= 1000
    % ----- LOW RANGE -----
    a(:,1) = Nstruct.a1(:);   b(:,1) = Nstruct.b1(:);
    a(:,2) = Ostruct.a1(:);   b(:,2) = Ostruct.b1(:);
    a(:,3) = NOstruct.a1(:);  b(:,3) = NOstruct.b1(:);
    a(:,4) = N2struct.a1(:);  b(:,4) = N2struct.b1(:);
    a(:,5) = O2struct.a1(:);  b(:,5) = O2struct.b1(:);

elseif T <= 6000
    % ----- HIGH RANGE -----
    a(:,1) = Nstruct.a2(:);   b(:,1) = Nstruct.b2(:);
    a(:,2) = Ostruct.a2(:);   b(:,2) = Ostruct.b2(:);
    a(:,3) = NOstruct.a2(:);  b(:,3) = NOstruct.b2(:);
    a(:,4) = N2struct.a2(:);  b(:,4) = N2struct.b2(:);
    a(:,5) = O2struct.a2(:);  b(:,5) = O2struct.b2(:);

else
    % ----- VERY HIGH RANGE -----
    a(:,1) = Nstruct.a3(:);   b(:,1) = Nstruct.b3(:);
    a(:,2) = Ostruct.a3(:);   b(:,2) = Ostruct.b3(:);
    a(:,3) = NOstruct.a3(:);  b(:,3) = NOstruct.b3(:);
    a(:,4) = N2struct.a3(:);  b(:,4) = N2struct.b3(:);
    a(:,5) = O2struct.a3(:);  b(:,5) = O2struct.b3(:);
end

end