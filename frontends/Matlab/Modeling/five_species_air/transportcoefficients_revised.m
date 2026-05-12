function [dT_drhos, dT_drhoe, Ds, hvec, mud, kappa, lambda, denom2, hs] = transportcoefficients_revised(T, rhos)
%TRANSPORTCOEFFICIENTS Transport and thermodynamic auxiliary quantities.
%   [dT_drhos, dT_drhoe, Ds, hvec, mud, kappa, lambda, denom2, hs] = ...
%       transportcoefficients(T, rhos)
%
%   T    : physical temperature [K]
%   rhos : physical species densities [kg/m^3]
%
%   Uses:
%     - NASA-9 thermodynamics for cp and h
%     - Gupta binary diffusion coefficients
%     - Blottner species viscosities
%     - Wilke mixture viscosity
%     - Eucken species thermal conductivity and Wilke-type mixing

[species_thermo_structs, Mw, RU] = thermodynamicsModels();
[species_structs, gupta_structs, ~, ~] = transport();

alpha = 1e4;
fT = elementaryfunctions(T);
lnT = fT(8);

rho_sum = sum(rhos);
Ys = rhos / rho_sum; % mass fractions

rho_tilde = rhos ./ Mw; % molar concentrations [mol/m^3]
rho_tsum = sum(rho_tilde);
Xs = rho_tilde / rho_tsum; % mole fractions

% Physical pressure
P = T * rho_tsum * RU;

ns = length(Mw);

% Species specific heat at constant pressure
cp = sym(zeros(ns, 1));
cpvec = sym(zeros(ns, 1));
for i = 1:ns
    Tsw1 = species_thermo_structs{i}.T1;
    Tsw2 = species_thermo_structs{i}.T2;
    fsw = switchfunctions(T, Tsw1, Tsw2, alpha);

    c1 = nasa9_cpcoeff(species_thermo_structs{i}.a1);
    c2 = nasa9_cpcoeff(species_thermo_structs{i}.a2);
    c3 = nasa9_cpcoeff(species_thermo_structs{i}.a3);

    cp1 = sum(c1 .* fT(1:7));
    cp2 = sum(c2 .* fT(1:7));
    cp3 = sum(c3 .* fT(1:7));

    cp(i) = fsw(1) * cp1 + fsw(2) * cp2 + fsw(3) * cp3;
    cpvec(i) = cp(i) * RU / Mw(i); % species cp on a mass basis
end

% Species enthalpy
hs = sym(zeros(ns, 1));
hvec = sym(zeros(ns, 1));
for i = 1:ns
    Tsw1 = species_thermo_structs{i}.T1;
    Tsw2 = species_thermo_structs{i}.T2;
    fsw = switchfunctions(T, Tsw1, Tsw2, alpha);

    c1 = nasa9_hcoeff(species_thermo_structs{i}.a1, species_thermo_structs{i}.b1);
    c2 = nasa9_hcoeff(species_thermo_structs{i}.a2, species_thermo_structs{i}.b2);
    c3 = nasa9_hcoeff(species_thermo_structs{i}.a3, species_thermo_structs{i}.b3);

    h1 = sum(c1 .* fT);
    h2 = sum(c2 .* fT);
    h3 = sum(c3 .* fT);

    hs(i) = fsw(1) * h1 + fsw(2) * h2 + fsw(3) * h3;
    hvec(i) = hs(i) * T * RU / Mw(i); % species enthalpy on a mass basis
end

cv = (cp - 1.0) * RU ./ Mw;
cvY = sum(cv .* Ys);
denom = sum(rhos) * cvY;
denom2 = denom;

% temperature derivatives
es = (hs - 1.0) * T * RU ./ Mw; % species internal energies on a mass basis
dT_drhos = -es / denom;
dT_drhoe = 1.0 / denom;

% Binary diffusion coefficients (Gupta correlation)
Dij = sym(zeros(ns, ns));
for i = 1:ns
    for j = 1:ns
        params = gupta_structs{i, j};
        expD = exp(params.D);
        t1 = params.A * lnT * lnT + params.B * lnT + params.C;
        Tterm = T^t1;
        Dij(i, j) = (expD * Tterm / P) * 10.1325;
    end
end
% The constant 10.1325 accounts for:
%   pressure conversion atm -> Pa: 1 atm = 101325 Pa
%   diffusion conversion cm^2/s -> m^2/s: 101325 * 1e-4 = 10.1325

% Mixture-averaged diffusion coefficients
Ds = sym(zeros(ns, 1));
for i = 1:ns
    denomD = sym(0);
    for j = 1:ns
        if i ~= j
            denomD = denomD + Xs(j) / Dij(i, j);
        end
    end
    Ds(i) = (1.0 - Xs(i)) / denomD;
end

% Species viscosities (Blottner correlation)
mus = sym(zeros(ns, 1));
for i = 1:ns
    params = species_structs{i};
    mus(i) = 0.1 * exp(params.A * lnT^2 + params.B * lnT + params.C);
end

% Wilke interaction coefficients
phi = sym(zeros(ns, 1));
for i = 1:ns
    for j = 1:ns
        if i == j
            phi(i) = phi(i) + Xs(j);
        else
            mu_ratio = mus(i) / mus(j);
            M_ratio = Mw(i) / Mw(j);
            tmp = 1.0 + sqrt(mu_ratio / sqrt(M_ratio));
            phi(i) = phi(i) + Xs(j) * tmp^2 / sqrt(8.0 * (1.0 + M_ratio));
        end
    end
end

% Mixture dynamic viscosity
mud = sum(mus .* Xs ./ phi);

% Species and mixture thermal conductivity
% Eucken relation at species level, then Wilke-type mixing.
lambda = mus .* (cpvec + 5/4 * RU ./ Mw);
kappa = sum(lambda .* Xs ./ phi);
end
