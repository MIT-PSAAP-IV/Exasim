function omega = kineticsource(T, rhos, alpha)
%KINETICSOURCE Kinetic source term for a five-species reacting air mixture.
%   omega = kineticsource(T, rhos)
%   T    : physical temperature [K]
%   rhos : physical species densities [kg/m^3], column or row vector
%
%   Returns omega [kg/(m^3 s)] for the species ordering used in
%   thermodynamicsModels / kinetics.

if nargin <= 2
  alpha = 1e3;
end

kinetics_params = kinetics;
[species_thermo_structs, Mw, RU] = thermodynamicsModels;

rho_tilde = rhos ./ Mw; % molar concentrations [mol/m^3]

A_r     = kinetics_params.A_r;
beta_r  = kinetics_params.beta_r;
theta_r = kinetics_params.theta_r;
nu_f_kj = kinetics_params.nu_f_kj;
nu_b_kj = kinetics_params.nu_b_kj;
alpha_jr = kinetics_params.alpha_jr;
P_atm   = kinetics_params.P_atm;
nr      = kinetics_params.nr;
ns      = kinetics_params.ns;

fT = elementaryfunctions(T);

% nondimensional Gibbs free energy corrected to the standard-state pressure
% contribution needed for the concentration-based equilibrium constant.
pressureTerm = log(P_atm / RU) - fT(8); % = log(P_atm/(RU*T))
Gformation = sym(zeros(1, ns));
for i = 1:ns
    Tsw1 = species_thermo_structs{i}.T1;
    Tsw2 = species_thermo_structs{i}.T2;
    fsw = switchfunctions(T, Tsw1, Tsw2, alpha);

    c1 = nasa9_Gcoeff(species_thermo_structs{i}.a1, species_thermo_structs{i}.b1);
    c2 = nasa9_Gcoeff(species_thermo_structs{i}.a2, species_thermo_structs{i}.b2);
    c3 = nasa9_Gcoeff(species_thermo_structs{i}.a3, species_thermo_structs{i}.b3);

    g1 = sum(c1 .* fT); % nondimensional Gibbs free energy, g_i/(Ru*T)
    g2 = sum(c2 .* fT);
    g3 = sum(c3 .* fT);

    Gformation(i) = fsw(1) * g1 + fsw(2) * g2 + fsw(3) * g3 - pressureTerm;
end

% natural logarithm of the forward reaction rates
lnkf_r = sym(zeros(1, nr));
for ir = 1:nr
    lnkf_r(ir) = log(A_r(ir)) + beta_r(ir) * fT(8) - theta_r(ir) * fT(6);
end

% nu_bf(k,r) = nu''_{k,r} - nu'_{k,r}
nu_bf = zeros(ns, nr);
for ir = 1:nr
    for k = 1:ns
        nu_bf(k, ir) = nu_b_kj(k, ir) - nu_f_kj(k, ir);
    end
end

% natural logarithm of the backward reaction rates
lnkb_r = sym(zeros(1, nr));
for ir = 1:nr
    lnkb_r(ir) = lnkf_r(ir);
    for k = 1:ns
        lnkb_r(ir) = lnkb_r(ir) + nu_bf(k, ir) * Gformation(k);
    end
end

kf_r = exp(lnkf_r);
kb_r = exp(lnkb_r);

% Third-body factors.
% For the present 5-reaction mechanism, reactions 1:3 are third-body
% dissociation/recombination reactions, while reactions 4:5 are exchange
% reactions without third-body enhancement.
Alpha_rs = zeros(nr, ns + 1);
Alpha_rs(1:3, 2:(ns + 1)) = alpha_jr';
Alpha_rs(4:nr, 1) = 1;
thirdbody_r = Alpha_rs * [sym(1); rho_tilde(:)];

% reaction progress rates
Rr = sym(zeros(1, nr));
for r = 1:nr
    Cf = sym(1);
    Cb = sym(1);
    for s = 1:ns
        Cf = Cf * rho_tilde(s)^nu_f_kj(s, r);
        Cb = Cb * rho_tilde(s)^nu_b_kj(s, r);
    end
    Rr(r) = (kf_r(r) * Cf - kb_r(r) * Cb) * thirdbody_r(r);
end

nuMw = zeros(ns, nr);
for k = 1:ns
    for r = 1:nr
        nuMw(k, r) = Mw(k) * (nu_b_kj(k, r) - nu_f_kj(k, r));
    end
end

% species source terms
omega = sym(zeros(ns, 1));
for k = 1:ns
    tmp = sym(0);
    for r = 1:nr
        tmp = tmp + nuMw(k, r) * Rr(r);
    end
    omega(k) = tmp;
end
end
