function [rhoE, p, emix] = energyFromSpecies(rho_species, T, v, alpha)
%ENERGYFROMSPECIES Compute mixture energy density, pressure, and internal energy
%from species densities, temperature, and velocity.
%
% Species order (fixed):
%   1 N, 2 O, 3 NO, 4 N2, 5 O2
%
% Inputs:
%   rho_species : 5x1 or 1x5 vector of species densities [kg/m^3]
%                 ordered as [N, O, NO, N2, O2]
%   T           : temperature [K], scalar
%   v           : velocity vector [u; v] or [u; v; w] [m/s]
%   alpha       : smoothing/sharpness parameter for switchfunctions
%                 (optional, default = 1e3)
%
% Outputs:
%   rhoE        : total energy density [J/m^3]
%   p           : pressure [Pa]
%   emix        : mixture internal energy [J/kg]
%
% Notes:
%   This function does NOT compute chemical equilibrium. It assumes the
%   input species densities are already given and computes the corresponding
%   mixture thermodynamic state consistently with equilibrate.m.

    if nargin < 4 || isempty(alpha)
        alpha = 1e3;
    end

    % -----------------------------
    % Input validation
    % % -----------------------------
    % validateattributes(rho_species, {'numeric'}, ...
    %     {'real','finite','vector','numel',5}, mfilename, 'rho_species', 1);
    % validateattributes(T, {'numeric'}, ...
    %     {'real','finite','scalar','positive'}, mfilename, 'T', 2);
    % validateattributes(v, {'numeric'}, ...
    %     {'real','finite','vector','nonempty'}, mfilename, 'v', 3);
    % validateattributes(alpha, {'numeric'}, ...
    %     {'real','finite','scalar','positive'}, mfilename, 'alpha', 4);

    rho_species = rho_species(:);   % 5x1 column vector
    v = v(:);                       % nv x 1 column vector

    % if any(rho_species < 0)
    %     error('rho_species must be nonnegative.');
    % end
    % 
    % if numel(v) ~= 2 && numel(v) ~= 3
    %     error('v must be a 2-component or 3-component velocity vector.');
    % end

    % -----------------------------
    % Thermodynamic data
    % -----------------------------
    [species_thermo_structs, Mw, Ru] = thermodynamicsModels();

    Mw = Mw(:);   % force 5x1
    validateattributes(Mw, {'numeric'}, ...
        {'real','finite','vector','numel',5,'positive'}, mfilename, 'Mw');

    Rspec = Ru ./ Mw;   % J/(kg*K)

    % -----------------------------
    % Mixture composition
    % -----------------------------
    rho = sum(rho_species);
    % if rho <= 0
    %     error('Total density must be positive.');
    % end

    Y = rho_species / rho;   % mass fractions

    nrho = rho_species ./ Mw;    % molar concentration up to common factor
    nrho_sum = sum(nrho);
    % if nrho_sum <= 0
    %     error('Total molar concentration must be positive.');
    % end

    x = nrho / nrho_sum;         % mole fractions
    Wmix = sum(x .* Mw);         % mixture molecular weight [kg/mol]
    Rmix = Ru / Wmix;            % mixture gas constant [J/(kg*K)]

    % -----------------------------
    % Pressure from ideal-gas EOS
    % -----------------------------
    p = rho * Rmix * T;
    % Equivalent form:
    % p = rho * sum(Y .* Rspec) * T;

    % -----------------------------
    % Species enthalpy/internal energy
    % -----------------------------
    fT = elementaryfunctions(T);
    fT = fT(:);   % enforce column vector
    
    hmolar = zeros(5,1);   % J/mol
    if class(T) == "sym"
      hmolar = sym(zeros(5,1));   % J/mol
    end

    for i = 1:5
        thermo_i = species_thermo_structs{i};

        Tsw1 = thermo_i.T1;
        Tsw2 = thermo_i.T2;
        fsw = switchfunctions(T, Tsw1, Tsw2, alpha);
        fsw = fsw(:);

        if numel(fsw) ~= 3
            error('switchfunctions must return a 3-component vector.');
        end

        c1 = nasa9_hcoeff(thermo_i.a1, thermo_i.b1); c1 = c1(:);
        c2 = nasa9_hcoeff(thermo_i.a2, thermo_i.b2); c2 = c2(:);
        c3 = nasa9_hcoeff(thermo_i.a3, thermo_i.b3); c3 = c3(:);

        if numel(c1) ~= numel(fT) || numel(c2) ~= numel(fT) || numel(c3) ~= numel(fT)
            error('Size mismatch: nasa9_hcoeff output must match elementaryfunctions(T).');
        end

        h1 = sum(c1 .* fT);
        h2 = sum(c2 .* fT);
        h3 = sum(c3 .* fT);

        H = fsw(1) * h1 + fsw(2) * h2 + fsw(3) * h3;   % dimensionless h/(Ru*T)
        hmolar(i) = H * Ru * T;                        % J/mol
    end

    hmass = hmolar ./ Mw;         % J/kg
    emass = hmass - Rspec * T;    % J/kg

    % -----------------------------
    % Mixture energies
    % -----------------------------
    emix = sum(Y .* emass);       % J/kg
    ke = 0.5 * sum(v.^2);         % J/kg
    E = emix + ke;                % J/kg
    rhoE = rho * E;               % J/m^3
end
