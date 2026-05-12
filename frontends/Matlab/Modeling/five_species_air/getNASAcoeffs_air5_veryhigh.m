function [a, b] = getNASAcoeffs_air5_veryhigh()
%GETNASACOEFFS_AIR5_VERYHIGH
% Returns NASA-9 very-high-temperature (typically 6000–20000 K)
% coefficients using the data from thermodynamicsModels().
%
% Output species order:
%   [N, O, NO, N2, O2]
%
% Coefficient format compatible with nasa9eval_G / nasa9eval_H:
%   a: 7x5  (a1..a7)
%   b: 2x5  (b1, b2)
%
% Very-high-temperature coefficients are taken from:
%   struct.a3 and struct.b3 in thermodynamicsModels()

% Load thermo database
[species_thermo_structs, ~, ~] = thermodynamicsModels();

% In thermodynamicsModels(): {N, O, NO, N2, O2}
Nstruct  = species_thermo_structs{1};
Ostruct  = species_thermo_structs{2};
NOstruct = species_thermo_structs{3};
N2struct = species_thermo_structs{4};
O2struct = species_thermo_structs{5};

% Allocate
a = zeros(7,5);
b = zeros(2,5);

% Desired output order: [N, O, NO, N2, O2]
a(:,1) = Nstruct.a3(:);   b(:,1) = Nstruct.b3(:);   % N
a(:,2) = Ostruct.a3(:);   b(:,2) = Ostruct.b3(:);   % O
a(:,3) = NOstruct.a3(:);  b(:,3) = NOstruct.b3(:);  % NO
a(:,4) = N2struct.a3(:);  b(:,4) = N2struct.b3(:);  % N2
a(:,5) = O2struct.a3(:);  b(:,5) = O2struct.b3(:);  % O2

end