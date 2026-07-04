
% argon_cylinder_quantities.m
%
% Compute freestream, transport, rarefaction, and derived quantities for
% hypersonic argon flow over a circular cylinder using the VHS viscosity model.
%
% Inputs:
%   U_inf       : freestream velocity [m/s]
%   M_inf       : freestream Mach number [-]
%   D           : cylinder diameter [m]
%   rho_inf     : freestream mass density [kg/m^3]
%   T_wall      : wall temperature [K]
%   omega       : VHS temperature exponent [-]
%   d_ref       : VHS reference molecular diameter [m]
%   T_ref       : VHS reference temperature [K]
%   Pr          : Prandtl number [-]
%
% Example corresponds approximately to the Mach-25, Kn=0.25 benchmark.

clear; clc;

%% -------------------------
% User inputs
%% -------------------------

U_inf   = 6585.0;             % freestream velocity [m/s]
M_inf   = 25.0;               % freestream Mach number [-]
D       = 12.0 * 0.0254;      % cylinder diameter [m], 12 inches
rho_inf = 1.127e-6;           % freestream density [kg/m^3]
T_wall  = 1500.0;             % wall temperature [K]

omega   = 0.734;              % VHS temperature exponent [-]
d_ref   = 3.595e-10;          % VHS reference diameter [m]
T_ref   = 1000.0;             % VHS reference temperature [K]
Pr      = 2.0/3.0;            % Prandtl number [-]

%% -------------------------
% Physical constants for argon
%% -------------------------

gamma = 5.0/3.0;              % ratio of specific heats for monatomic gas [-]
Ru    = 8.31446261815324;     % universal gas constant [J/(mol K)]
NA    = 6.02214076e23;        % Avogadro constant [1/mol]
kB    = 1.380649e-23;         % Boltzmann constant [J/K]

M_molar = 39.948e-3;          % argon molar mass [kg/mol]
m       = M_molar / NA;       % argon molecular mass [kg]
R       = Ru / M_molar;       % specific gas constant for argon [J/(kg K)]

%% -------------------------
% Freestream thermodynamic quantities
%% -------------------------

a_inf = U_inf / M_inf;                  % speed of sound [m/s]
T_inf = a_inf^2 / (gamma * R);          % freestream temperature [K]
p_inf = rho_inf * R * T_inf;            % freestream pressure [Pa]

cv = R / (gamma - 1.0);                 % constant-volume specific heat [J/(kg K)]
cp = gamma * R / (gamma - 1.0);         % constant-pressure specific heat [J/(kg K)]

%% -------------------------
% VHS viscosity model
%% -------------------------
% Reference viscosity:
%
% mu_ref = 15 sqrt(pi m kB T_ref)
%          -----------------------------------------------
%          2 pi d_ref^2 (5 - 2 omega)(7 - 2 omega)

mu_ref = 15.0 * sqrt(pi * m * kB * T_ref) / ...
         (2.0 * pi * d_ref^2 * (5.0 - 2.0*omega) * (7.0 - 2.0*omega));

mu_inf = mu_ref * (T_inf / T_ref)^omega;   % freestream viscosity [Pa s]
mu_wall = mu_ref * (T_wall / T_ref)^omega; % wall-temperature viscosity [Pa s]

nu_inf = mu_inf / rho_inf;                 % freestream kinematic viscosity [m^2/s]

%% -------------------------
% Thermal conductivity
%% -------------------------

kappa_inf  = mu_inf  * cp / Pr;            % freestream thermal conductivity [W/(m K)]
kappa_wall = mu_wall * cp / Pr;            % wall-temperature thermal conductivity [W/(m K)]

%% -------------------------
% Reynolds number and flow quantities
%% -------------------------

Re_D = rho_inf * U_inf * D / mu_inf;       % Reynolds number based on diameter [-]

q_inf = 0.5 * rho_inf * U_inf^2;           % dynamic pressure [Pa]
rhoU2 = rho_inf * U_inf^2;                 % momentum flux [Pa]

%% -------------------------
% Mean free path and Knudsen number
%% -------------------------
% Viscosity-based mean free path consistent with the VHS viscosity:
%
% lambda = mu/rho * sqrt(pi/(2 R T))
%
% This gives a VHS/viscosity-consistent mean free path.
% For the benchmark paper, Kn was reported using a hard-sphere mean-free-path
% calculation based on freestream conditions and cylinder diameter.

lambda_visc = (mu_inf / rho_inf) * sqrt(pi / (2.0 * R * T_inf));
Kn_visc = lambda_visc / D;

% Hard-sphere mean free path using d_ref directly:
%
% lambda_hs = m/(sqrt(2) pi d_ref^2 rho)
%
% Note: this uses d_ref at T_ref directly. Some DSMC/VHS conventions use
% an effective temperature-dependent collision diameter; therefore this
% value may differ from the viscosity-consistent value by a constant factor.

lambda_hs = m / (sqrt(2.0) * pi * d_ref^2 * rho_inf);
Kn_hs = lambda_hs / D;

%% -------------------------
% Mach/Re/Kn consistency relation
%% -------------------------

Re_from_Mach_Kn_visc = (M_inf / Kn_visc) * sqrt(gamma*pi/2.0);

%% -------------------------
% Print results
%% -------------------------

fprintf('\nARGON HYPERSONIC CYLINDER QUANTITIES\n');
fprintf('------------------------------------\n\n');

fprintf('Input quantities:\n');
fprintf('  U_inf       = %.10e m/s\n', U_inf);
fprintf('  M_inf       = %.10e\n', M_inf);
fprintf('  D           = %.10e m\n', D);
fprintf('  rho_inf     = %.10e kg/m^3\n', rho_inf);
fprintf('  T_wall      = %.10e K\n', T_wall);
fprintf('  omega       = %.10e\n', omega);
fprintf('  d_ref       = %.10e m\n', d_ref);
fprintf('  T_ref       = %.10e K\n', T_ref);
fprintf('  Pr          = %.10e\n', Pr);

fprintf('\nGas constants:\n');
fprintf('  gamma       = %.10e\n', gamma);
fprintf('  M_molar     = %.10e kg/mol\n', M_molar);
fprintf('  m           = %.10e kg\n', m);
fprintf('  R           = %.10e J/(kg K)\n', R);
fprintf('  Ru          = %.10e J/(mol K)\n', Ru);
fprintf('  kB          = %.10e J/K\n', kB);
fprintf('  NA          = %.10e 1/mol\n', NA);

fprintf('\nFreestream thermodynamic quantities:\n');
fprintf('  a_inf       = %.10e m/s\n', a_inf);
fprintf('  T_inf       = %.10e K\n', T_inf);
fprintf('  p_inf       = %.10e Pa\n', p_inf);
fprintf('  cv          = %.10e J/(kg K)\n', cv);
fprintf('  cp          = %.10e J/(kg K)\n', cp);

fprintf('\nVHS transport quantities:\n');
fprintf('  mu_ref      = %.10e Pa s\n', mu_ref);
fprintf('  mu_inf      = %.10e Pa s\n', mu_inf);
fprintf('  mu_wall     = %.10e Pa s\n', mu_wall);
fprintf('  nu_inf      = %.10e m^2/s\n', nu_inf);

fprintf('\nThermal transport quantities:\n');
fprintf('  kappa_inf   = %.10e W/(m K)\n', kappa_inf);
fprintf('  kappa_wall  = %.10e W/(m K)\n', kappa_wall);

fprintf('\nDerived flow quantities:\n');
fprintf('  Re_D        = %.10e\n', Re_D);
fprintf('  q_inf       = %.10e Pa\n', q_inf);
fprintf('  rhoU2       = %.10e Pa\n', rhoU2);

fprintf('\nRarefaction quantities:\n');
fprintf('  lambda_visc = %.10e m\n', lambda_visc);
fprintf('  Kn_visc     = %.10e\n', Kn_visc);
fprintf('  lambda_hs   = %.10e m\n', lambda_hs);
fprintf('  Kn_hs       = %.10e\n', Kn_hs);

fprintf('\nConsistency check:\n');
fprintf('  Re from M and Kn_visc = %.10e\n', Re_from_Mach_Kn_visc);
fprintf('  Re_D                  = %.10e\n', Re_D);
fprintf('\n');

%% -------------------------
% Store results in a structure
%% -------------------------

Q = struct();

Q.U_inf = U_inf;
Q.M_inf = M_inf;
Q.D = D;
Q.rho_inf = rho_inf;
Q.T_wall = T_wall;

Q.gamma = gamma;
Q.M_molar = M_molar;
Q.m = m;
Q.R = R;
Q.Ru = Ru;
Q.NA = NA;
Q.kB = kB;
Q.Pr = Pr;

Q.omega = omega;
Q.d_ref = d_ref;
Q.T_ref = T_ref;

Q.a_inf = a_inf;
Q.T_inf = T_inf;
Q.p_inf = p_inf;
Q.cv = cv;
Q.cp = cp;

Q.mu_ref = mu_ref;
Q.mu_inf = mu_inf;
Q.mu_wall = mu_wall;
Q.nu_inf = nu_inf;

Q.kappa_inf = kappa_inf;
Q.kappa_wall = kappa_wall;

Q.Re_D = Re_D;
Q.q_inf = q_inf;
Q.rhoU2 = rhoU2;

Q.lambda_visc = lambda_visc;
Q.Kn_visc = Kn_visc;
Q.lambda_hs = lambda_hs;
Q.Kn_hs = Kn_hs;

Q.Re_from_Mach_Kn_visc = Re_from_Mach_Kn_visc;

% Optional: save output structure.
% save('argon_cylinder_quantities_output.mat','Q');
