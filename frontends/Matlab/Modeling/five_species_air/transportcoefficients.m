function [dT_drhos, dT_drhoe, Ds, hvec, mud, kappa, lambda, denom2, hs] = transportcoefficients(T, rhos)
% physical temperature: T
% physical species densities: rho_s

[species_thermo_structs, Mw, RU] = thermodynamicsModels();
[species_structs, gupta_structs, gupta_mu_structs, ~] = transport();
Tsw1 = species_thermo_structs{1}.T1;
Tsw2 = species_thermo_structs{1}.T2;

alpha = 1e4;
fT = elementaryfunctions(T);
fsw = switchfunctions(T, Tsw1, Tsw2, alpha);

rho_sum = sum(rhos);
Ys = rhos/rho_sum; % Y: the mass fractions of species

rho_tilde = rhos ./ Mw; % the molar densities of species
rho_tsum = sum(rho_tilde); 
Xs = rho_tilde/rho_tsum; % X: the mole fractions of species

% Physical pressure
P = T * sum(rhos ./ Mw) * RU; % p_dim

ns = length(Mw);

% Species specific heat at constant pressure
cp = sym(zeros(ns,1));
cpvec = sym(zeros(ns,1));
for i = 1:ns
  c1 = nasa9_cpcoeff(species_thermo_structs{i}.a1);
  c2 = nasa9_cpcoeff(species_thermo_structs{i}.a2);
  c3 = nasa9_cpcoeff(species_thermo_structs{i}.a3);
  h1 = sum(c1.*fT(1:7));
  h2 = sum(c2.*fT(1:7));
  h3 = sum(c3.*fT(1:7));  
  cp(i) = fsw(1)*h1 + fsw(2)*h2 + fsw(3)*h3;
  cpvec(i) = cp(i) * RU / Mw(i); % getCpsMass
end

% Species enthalpy
hs = sym(zeros(ns,1));
hvec = sym(zeros(ns,1));
for i = 1:ns
  c1 = nasa9_hcoeff(species_thermo_structs{i}.a1, species_thermo_structs{i}.b1);
  c2 = nasa9_hcoeff(species_thermo_structs{i}.a2, species_thermo_structs{i}.b2);
  c3 = nasa9_hcoeff(species_thermo_structs{i}.a3, species_thermo_structs{i}.b3);
  h1 = sum(c1.*fT);
  h2 = sum(c2.*fT);
  h3 = sum(c3.*fT);  
  hs(i) = fsw(1)*h1 + fsw(2)*h2 + fsw(3)*h3;
  hvec(i) = hs(i) * T * RU / Mw(i); % h_vec  
end

cv = (cp - 1.0) * RU ./ Mw; % getCvsMass
cvY = sum(cv.*Ys);          % mixtureFrozenCvMass
denom = sum(rhos) * cvY;    % denom
denom2 = denom;

% temperature derivatives
es = (hs - 1.0)*T*RU ./ Mw; % e_i
dT_drhos = -es/denom; % dT_drho_i_dim
dT_drhoe = 1.0/denom; % dT_drhoe_dim

% Binary Diffusion Coefficients
Dij = sym(zeros(ns, ns));
lnT = fT(8);
for i = 1:ns
  for j = 1:ns    
    params = gupta_structs{i,j};
    expD = exp(params.D);
    t1 = params.A * lnT*lnT + params.B * lnT + params.C;    
    Tterm = T^t1;                            % Tterm in evalCurveFit_guptaDij
    Dij(i,j) = (expD * Tterm / P) * 10.1325; % Dij in averageDiffusionCoeffs   
  end
end
% The constant 10.1325 accounts for
% Pressure conversion atm → Pa: 1atm = 101325 Pa
% Diffusion unit conversion cm²/s → m²/s: 101325 * 10^{-4} = 10.1325

% Mixture-Averaged Diffusion Coefficients
Ds = sym(zeros(ns,1)); 
for i = 1:ns
  denom = sym(0); % denom in averageDiffusionCoeffs
  for j = 1:ns
    if i ~= j
      denom = denom + Xs(j) / Dij(i,j);
    end
  end
  Ds(i) = (1.0 - Xs(i)) / denom; % D_vec
end    

% Species Viscosity
mus = sym(zeros(ns,1));
for i = 1:ns
  %params = gupta_mu_structs{i};
  params = species_structs{i};
  t1 = params.A * lnT + params.B;
  Tterm = T^t1;
  expC = exp(params.C);
  mus(i) = 0.1 * expC * Tterm; % mu_i
end

% The interaction coefficients of species
phi = sym(zeros(ns,1));
for i = 1:ns
  for j = 1:ns
    if i == j
      phi(i) = phi(i) + Xs(j);      
    else
      mu_ratio = mus(i) / mus(j);
      M_ratio = Mw(i) / Mw(j);
      tmp = 1.0 + sqrt(mu_ratio / sqrt(M_ratio));  % is this correct?    
      phi(i) = phi(i) + Xs(j) * tmp^2 / sqrt(8.0 *(1.0 + M_ratio)); % phi_i
    end
  end
end

% Mixture Dynamic Viscosity
mud =  sum( mus .* Xs ./ phi); % mu_d_dim

% Mixture Thermal Conductivity
lambda = mus .*  (cpvec + 5/4 * RU./Mw); % lambda_i
kappa =  sum( lambda .* Xs ./ phi); % kappa_dim
