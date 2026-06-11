function info = mixtureFromSpecies(rho_species, T, v)
% mixtureFromSpecies Initialize reacting-air mixture state from species densities.
%
% Species order (FIXED everywhere in this function):
%   1 N, 2 O, 3 NO, 4 N2, 5 O2
%
% Inputs:
%   rho_species : 5x1 or 1x5 vector of species densities [kg/m^3]
%                 ordered as [N, O, NO, N2, O2]
%   T           : temperature [K]
%   v           : velocity vector [u; v] or [u; v; w] [m/s]
%
% Outputs:
%   info        : struct with fields
%       .rho            total density [kg/m^3]
%       .rho_species    species densities [kg/m^3]
%       .Y              mass fractions
%       .x              mole fractions
%       .Wmix           mixture molecular weight [kg/mol]
%       .Rspec          species gas constants [J/(kg*K)]
%       .Rmix           mixture gas constant [J/(kg*K)]
%       .p              pressure [Pa]
%       .emass          species internal energies [J/kg]
%       .hmass          species enthalpies [J/kg]
%       .hmolar         species enthalpies [J/mol]
%       .emix           mixture internal energy [J/kg]
%       .E              mixture total specific energy [J/kg]
%       .rhoE           total energy density [J/m^3]
%       .rhov           momentum density
%
% Notes:
%   This function does NOT compute chemical equilibrium. It assumes the
%   input species densities are already given and computes the corresponding
%   thermodynamic mixture state consistently with equilibrate.m.

    % ---- constants ----
    Ru = 8.31446261815324; % J/(mol*K)

    % ---- basic checks ----
    assert(isnumeric(T) && isscalar(T) && T > 0, 'T must be a positive scalar.');
    assert(isnumeric(rho_species) && numel(rho_species) == 5, ...
        'rho_species must be a 5-component vector.');
    assert(all(rho_species(:) >= 0), 'rho_species must be nonnegative.');

    rho_species = rho_species(:).'; % row vector, [N O NO N2 O2]

    if nargin < 3 || isempty(v)
        v = 0;
    end
    v = v(:); % column vector

    % ---- thermo data ----
    [thermo.a, thermo.b] = getNASAcoeffs_air5(T);

    thermo.W = [ ...
        14.0067e-3, ... % N
        15.9994e-3, ... % O
        30.0061e-3, ... % NO
        28.0134e-3, ... % N2
        31.9988e-3  ... % O2
    ];

    assert(all(size(thermo.a) == [7 5]), 'thermo.a must be 7x5');
    assert(all(size(thermo.b) == [2 5]), 'thermo.b must be 2x5');

    W = thermo.W(:).';         % kg/mol
    Rspec = Ru ./ W;           % J/(kg*K)

    % ---- mixture composition ----
    rho = sum(rho_species);
    assert(rho > 0, 'Total density must be positive.');

    Y = rho_species / rho;     % mass fractions

    % mole fractions from species densities
    nrho = rho_species ./ W;   % mol/m^3 up to factor
    nrho_sum = sum(nrho);
    assert(nrho_sum > 0, 'Total molar concentration must be positive.');

    x = nrho / nrho_sum;       % mole fractions
    Wmix = sum(x .* W);        % kg/mol
    Rmix = Ru / Wmix;          % J/(kg*K)

    % ---- pressure from EOS ----
    p = rho * Rmix * T;
    % equivalent: p = rho * sum(Y .* Rspec) * T;

    % ---- species enthalpy/internal energy from NASA-9 ----
    hmolar = zeros(1,5);
    for i = 1:5
        H = nasa9eval_H(T, thermo.a(:,i), thermo.b(:,i)); % h^o/(Ru*T)
        hmolar(i) = H * Ru * T;                           % J/mol
    end

    hmass = hmolar ./ W;        % J/kg
    emass = hmass - Rspec * T;  % J/kg

    % ---- mixture energies ----
    emix = sum(Y .* emass);     % J/kg
    ke = 0.5 * sum(v.^2);       % J/kg
    E = emix + ke;              % total specific energy [J/kg]
    rhoE = rho * E;             % total energy density [J/m^3]
    rhov = rho * v;             % momentum density

    % ---- output ----
    info = struct();
    info.rho = rho;
    info.rho_species = rho_species;
    info.T = T;
    info.v = v;
    info.Y = Y;                 % mass fractions [N O NO N2 O2]
    info.x = x;                 % mole fractions [N O NO N2 O2]
    info.Wmix = Wmix;           % kg/mol
    info.Rspec = Rspec;         % J/(kg*K)
    info.Rmix = Rmix;           % J/(kg*K)
    info.p = p;                 % Pa
    info.emass = emass;         % J/kg
    info.hmass = hmass;         % J/kg
    info.hmolar = hmolar;       % J/mol
    info.emix = emix;           % J/kg
    info.E = E;                 % J/kg
    info.rhoE = rhoE;           % J/m^3
    info.rhov = rhov;           % kg/(m^2 s) or kg/(m^2 s) per component
end

% =========================================================================
% NASA-9 evaluators (consistent with equilibrate.m)
% =========================================================================
function H = nasa9eval_H(T, a, b)
    T2 = T*T; T3 = T2*T; T4 = T3*T;
    Tinv = 1.0/T; logT = log(T);
    H = (-a(1) * 1.0/(T2) + a(2) * logT * Tinv + a(3) ...
        + a(4) * T/2 + a(5) * T2/3.0 + a(6) * T3/4.0 + a(7) * T4/5.0 + b(1)/T);
end