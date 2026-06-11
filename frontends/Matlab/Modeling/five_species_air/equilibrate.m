function info = equilibrate(p, T, v)
%INIT_REACTING5_FROM_IDEAL  Initialize 5-species reacting air from ideal-gas solution.
%
% Species order (FIXED everywhere in this function):
%   1 N, 2 O, 3 NO, 4 N2, 5 O2
%
% Inputs:
%   rho    : density
%   rhou   : momentum vector [rho*u; rho*v; (rho*w)]  (size 2 or 3)
%   rhoE   : total energy density (rho*E) from ideal run (not strictly required)
%   p      : pressure (local, from ideal run)
%   T      : temperature (local, from ideal run)
%   thermo : struct with fields
%       .a   : 7x5 NASA-9 'a' coefficients (columns per species, in order [N O NO N2 O2])
%       .b   : 2x5 NASA-9 'b' coefficients (columns per species, in order [N O NO N2 O2])
%       .W   : 1x5 molecular weights [kg/mol], in order [N O NO N2 O2]
%       .p0  : standard pressure (Pa), default 101325
%       .rON : O-atom to N-atom ratio in mixture, default 0.21/0.79
%
% Outputs:
%   rho_species : 1x5 species densities [rho_N rho_O rho_NO rho_N2 rho_O2]
%   rhoE_new    : updated rhoE consistent with (rho, rhou, T, Y)
%   info        : struct with fields x (mole frac), Y (mass frac), lnKp, iters, converged, p_from_eos

    % ---- constants ----
    Ru = 8.31446261815324; % J/(mol*K)

    % Species order: [N, O, NO, N2, O2]
    [thermo.a, thermo.b] = getNASAcoeffs_air5(T);
    
    thermo.W = [ ...
        14.0067e-3, ... % N
        15.9994e-3  ... % O  
        30.0061e-3, ... % NO
        28.0134e-3, ... % N2
        31.9988e-3, ... % O2    
    ];
        
    thermo.p0  = 101325.0;        % 1 atm
    thermo.rON = 0.21/0.79;       % O-atom/N-atom ratio of air (adjust if desired)
    
    % ---- check thermo ----
    assert(isfield(thermo,'a') && all(size(thermo.a)==[7 5]), 'thermo.a must be 7x5');
    assert(isfield(thermo,'b') && all(size(thermo.b)==[2 5]), 'thermo.b must be 2x5');
    assert(isfield(thermo,'W') && numel(thermo.W)==5, 'thermo.W must be 1x5 [kg/mol]');
    if ~isfield(thermo,'p0'),  thermo.p0  = 101325.0; end
    if ~isfield(thermo,'rON'), thermo.rON = 0.21/0.79; end

    p0  = thermo.p0;
    rON = thermo.rON;
    W   = thermo.W(:).'; % row, order [N O NO N2 O2]

    % ---- compute Kp(T) from NASA-9 Gibbs (dimensionless g/RT) ----
    % G order: [N O NO N2 O2]
    G = zeros(1,5);
    for i = 1:5
        G(i) = nasa9eval_G(T, thermo.a(:,i), thermo.b(:,i)); % g_i^o/(Ru*T)
    end

    % Reaction stoichiometry nu in order [N O NO N2 O2]
    % R1: N2 -> 2N      : 2N - N2 = 0        nu = [ +2, 0, 0, -1, 0 ]
    % R2: O2 -> 2O      : 2O - O2 = 0        nu = [  0,+2, 0,  0,-1 ]
    % R3: NO -> N + O   : N + O - NO = 0     nu = [ +1,+1,-1,  0, 0 ]
    nu1 = [ 2  0  0 -1  0];
    nu2 = [ 0  2  0  0 -1];
    nu3 = [ 1  1 -1  0  0];

    % ln Kp = - sum_i (nu_i * G_i)   with G_i = g_i^o/(Ru*T)
    lnK1 = -dot(nu1, G);
    lnK2 = -dot(nu2, G);
    lnK3 = -dot(nu3, G);

    % ---- Newton solve in log-space (softmax) for mole fractions x ----
    % x order (same): [N O NO N2 O2]
    opts.maxit   = 60;
    opts.tolF    = 1e-12;
    opts.tolX    = 1e-12;
    opts.verbose = false;

    % Initial guess (mostly N2/O2, tiny NO/N/O) but in [N O NO N2 O2] order:
    x0 = [1e-12, 1e-12, 1e-12, 0.79, 0.21]';
    x0 = x0 / sum(x0);

    eta = log(x0(1:4) / x0(5)); % eta5 fixed to 0 (species 5 is O2)
    [eta, x, iters, converged] = newton_softmax_equil(eta, p, p0, rON, lnK1, lnK2, lnK3, opts);

    % ---- convert to mass fractions Y and species densities ----
    Wmix = sum(x .* W);
    Y = (x .* W) / Wmix;          % mass fractions, order [N O NO N2 O2]

    Rspec  = Ru ./ W;             % J/(kg*K), order [N O NO N2 O2]

    % ---- update rhoE using NASA-9 enthalpy -> internal energy ----
    hmolar = zeros(1,5);
    for i = 1:5
        H = nasa9eval_H(T, thermo.a(:,i), thermo.b(:,i)); % h^o/(Ru*T)
        hmolar(i) = H * Ru * T;                           % J/mol
    end
    hmass = hmolar ./ W;        % J/kg
    emass = hmass - Rspec * T;  % J/kg        
    emix   = sum(Y .* emass); % e = RU * T * (sum(Y./W .*  H) - sum(Y./W))
    
    % ---- info ----
    info = struct();
    info.eta = eta;
    info.rho = p/(sum(Y .* Rspec) * T);
    info.rho_species = info.rho .* Y;
    info.x = x;                       % mole fractions [N O NO N2 O2]
    info.Y = Y;                       % mass fractions [N O NO N2 O2]
    info.Wmix = Wmix;                 % kg/mol
    info.emix = emix;                 % J/kg
    info.emass = emass;               % J/kg
    info.hmass = hmass;               % J/kg
    info.hmolar = hmolar;             % J/molar
    info.lnKp = [lnK1 lnK2 lnK3];
    info.iters = iters;
    info.converged = converged;
    if nargin > 2
      ke = 0.5*sum(v.*v);
      info.E = emix + ke;
      info.rhoE = info.rho * (emix + ke);
      info.rhov = info.rho * v;
    end
end

% =========================================================================
% Robust Newton in log-space using softmax parameterization
% =========================================================================
function [eta, x, iters, converged] = newton_softmax_equil(eta, p, p0, rON, lnK1, lnK2, lnK3, opts)
    maxit = opts.maxit; tolF = opts.tolF; tolX = opts.tolX;

    converged = false;
    iters = 0;

    for k = 1:maxit
        iters = k;

        [x, dx_deta] = softmax_with_jac(eta); % x:1x5 in order [N O NO N2 O2]
        
        F = resid_equil(x, p, p0, rON, lnK1, lnK2, lnK3);
        nF = norm(F,2);
        if nF < tolF
            converged = true;
            return;
        end

        J = jac_equil(x, dx_deta, rON);

        d = J \ F;
        step = -d;

        if norm(step,2) < tolX
            converged = true;
            return;
        end

        % Backtracking line search
        alpha = 1.0;
        nF0 = nF;
        for ls = 1:25
            eta_try = eta + alpha * step;
            [x_try, ~] = softmax_with_jac(eta_try);
            F_try = resid_equil(x_try, p, p0, rON, lnK1, lnK2, lnK3);
            if norm(F_try,2) <= (1 - 1e-4*alpha) * nF0
                eta = eta_try;
                break;
            end
            alpha = 0.5 * alpha;
        end
        
        if alpha < 1e-7
            eta = eta + 1e-3 * step;
        end
    end
    
    [x, ~] = softmax_with_jac(eta);
end

function F = resid_equil(x, p, p0, rON, lnK1, lnK2, lnK3)
    % x order: [N O NO N2 O2]
    xN  = x(1); xO  = x(2); xNO = x(3); xN2 = x(4); xO2 = x(5);

    % Element ratio constraint: O_atoms - rON * N_atoms = 0
    % N_atoms = xN + xNO + 2 xN2
    % O_atoms = xO + xNO + 2 xO2
    Na = xN + xNO + 2*xN2;
    Oa = xO + xNO + 2*xO2;
    F1 = Oa - rON * Na;

    ln_p_over_p0 = log(p / p0);
    epsx = 1e-300;

    ln_xN  = log(max(xN , epsx));
    ln_xO  = log(max(xO , epsx));
    ln_xNO = log(max(xNO, epsx));
    ln_xN2 = log(max(xN2, epsx));
    ln_xO2 = log(max(xO2, epsx));

    % R1: N2 <-> 2N : 2 ln(pN/p0) - ln(pN2/p0) - lnK1 = 0
    F2 = (2*(ln_xN  + ln_p_over_p0) - (ln_xN2 + ln_p_over_p0)) - lnK1;

    % R2: O2 <-> 2O
    F3 = (2*(ln_xO  + ln_p_over_p0) - (ln_xO2 + ln_p_over_p0)) - lnK2;

    % R3: NO <-> N + O : ln(pN/p0)+ln(pO/p0)-ln(pNO/p0)-lnK3 = 0
    F4 = ((ln_xN + ln_p_over_p0) + (ln_xO + ln_p_over_p0) - (ln_xNO + ln_p_over_p0)) - lnK3;

    F = [F1; F2; F3; F4];
end

function J = jac_equil(x, dx_deta, rON)
    % x order: [N O NO N2 O2], dx_deta: 5x4

    xN  = x(1); xO  = x(2); xNO = x(3); xN2 = x(4); xO2 = x(5);

    % F1 = (xO + xNO + 2 xO2) - rON*(xN + xNO + 2 xN2)
    dF1_dx = [-rON, 1, (1 - rON), -2*rON, 2];

    % F2 = 2 ln xN - ln xN2 + const
    dF2_dx = [2*(1/xN), 0, 0, -(1/xN2), 0];

    % F3 = 2 ln xO - ln xO2 + const
    dF3_dx = [0, 2*(1/xO), 0, 0, -(1/xO2)];

    % F4 = ln xN + ln xO - ln xNO + const
    dF4_dx = [(1/xN), (1/xO), -(1/xNO), 0, 0];

    J = zeros(4,4);
    J(1,:) = dF1_dx * dx_deta;
    J(2,:) = dF2_dx * dx_deta;
    J(3,:) = dF3_dx * dx_deta;
    J(4,:) = dF4_dx * dx_deta;
end

function [x, dx_deta] = softmax_with_jac(eta)
    % eta: 1x4, eta5 fixed to 0 (species 5 = O2)
    z = [eta(:).' 0.0];
    z = z - max(z);
    ex = exp(z);
    S  = sum(ex);
    x  = ex / S;

    dx_deta = zeros(5,4);
    for k = 1:4
        for i = 1:5
            dx_deta(i,k) = x(i) * ((i==k) - x(k));
        end
    end
end

% =========================================================================
% NASA-9 evaluators (dimensionless H/RT and G/RT)
% =========================================================================
function G = nasa9eval_G(T, a, b)
    T2 = T*T; T3 = T2*T; T4 = T3*T;
    Tinv = 1.0/T; logT = log(T);
    G = (-a(1) * 1.0/(2*T2) + a(2) * (logT + 1.0) * Tinv + a(3) * (1 - logT) ...
        - a(4) * T/2 - a(5) * T2/6.0 - a(6) * T3/12.0 - a(7) * T4/20.0 + b(1)/T - b(2));
end

function H = nasa9eval_H(T, a, b)
    T2 = T*T; T3 = T2*T; T4 = T3*T;
    Tinv = 1.0/T; logT = log(T);
    H = (-a(1) * 1.0/(T2) + a(2) * logT * Tinv + a(3) ...
        + a(4) * T/2 + a(5) * T2/3.0 + a(6) * T3/4.0 + a(7) * T4/5.0 + b(1)/T);
end