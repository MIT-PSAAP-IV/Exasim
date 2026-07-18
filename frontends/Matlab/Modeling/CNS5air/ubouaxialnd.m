function ub = ubouaxialnd(u, q, w, v, x, t, mu, eta, uh, n, tau, uinf, param, ib)
%UBOUAXIALND LDG exterior states for axial five-species reacting air.
%
% Boundary columns follow fbouhdgaxialnd.m:
%   1 inflow, 2 outflow, 3 isothermal no-slip wall, 4 slip/axis symmetry,
%   5 zero-gradient, 6 noncatalytic wall, 7 supercatalytic wall,
%   8 partial-catalysis wall, 9 stoichiometric partial-catalysis wall.

    %#ok<*INUSD>
    ns = 5;
    nd = numel(x);
    ncu = ns + nd + 1;

    u   = u(:);
    w   = w(:);
    mu  = mu(:);
    eta = eta(:);
    n   = n(:);

    if numel(u) ~= ncu
        error('ubouaxialnd:InvalidUSize', 'u must have length %d, but got %d.', ncu, numel(u));
    end
    if numel(w) < 1
        error('ubouaxialnd:InvalidWSize', 'w must contain at least one entry.');
    end
    if numel(mu) < 12
        error('ubouaxialnd:InvalidMuSize', 'mu must contain at least 12 entries.');
    end
    if numel(eta) < ncu + 2*ns
        error('ubouaxialnd:InvalidEtaSize', ...
            'eta must contain [ub(%d); Ycat(%d); gamma(%d)].', ncu, ns, ns);
    end

    irho  = 1:ns;
    irhou = ns + (1:nd);
    irhoE = ns + nd + 1;

    rho_scale  = mu(1);
    rhoe_scale = mu(3);
    T_scale    = mu(4);
    T_wall     = mu(12);

    ub_in = eta(1:ncu);
    Y_i_cat = eta(ncu + (1:ns));

    [~, Mw, RU] = thermodynamicsModels();
    Mw = Mw(:);

    v_wall = 0*n;

    % 2,5) Outflow and zero-gradient use the interior state as the LDG trace.
    ub_out = u;
    ub_grad = u;

    % 3) Isothermal no-slip wall: preserve species, zero velocity, reset rhoE.
    rho_i_wall_dim = u(irho) * rho_scale;
    [rhoE_dim_wall, ~, ~] = energyFromSpecies(rho_i_wall_dim, T_wall, v_wall);
    rhoE_wall = rhoE_dim_wall / rhoe_scale;

    ub_iso = u;
    ub_iso(irhou) = 0;
    ub_iso(irhoE) = rhoE_wall;

    % 4) Slip wall / axis symmetry: remove only normal momentum.
    m = u(irhou);
    mn = m.' * n;
    ub_slip = u;
    ub_slip(irhou) = m - mn .* n;

    % Thermodynamic state used by catalytic wall states.
    T_flow = w(1) * T_scale;
    rho_i_flow = u(irho) * rho_scale;
    rho_flow = sum(rho_i_flow);
    p_flow = pressure(T_flow, rho_i_flow, Mw);

    % 6) Noncatalytic wall: preserve interior mass fractions at wall T.
    Smix = sum(rho_i_flow ./ Mw) / rho_flow;
    rho_noncat = p_flow / (RU * T_wall * Smix);
    rho_i_noncat = (rho_noncat / rho_flow) * rho_i_flow;
    [rhoE_dim_noncat, ~, ~] = energyFromSpecies(rho_i_noncat, T_wall, v_wall);

    ub_noncat = 0*u;
    ub_noncat(irho) = rho_i_noncat / rho_scale;
    ub_noncat(irhou) = 0;
    ub_noncat(irhoE) = rhoE_dim_noncat / rhoe_scale;

    % 7) Supercatalytic wall: prescribe wall mass fractions at p_flow,T_wall.
    Y_i_cat = Y_i_cat(:);
    Y_i_cat = Y_i_cat / sum(Y_i_cat);
    rho_cat = densityFromMassFractions(T_wall, p_flow, Y_i_cat, Mw);
    rho_i_cat = rho_cat * Y_i_cat;
    [rhoE_dim_cat, ~, ~] = energyFromSpecies(rho_i_cat, T_wall, v_wall);

    ub_cat = 0*u;
    ub_cat(irho) = rho_i_cat / rho_scale;
    ub_cat(irhou) = 0;
    ub_cat(irhoE) = rhoE_dim_cat / rhoe_scale;

    % 8-9) Catalytic-flux walls use the isothermal trace. fbouaxialnd
    % replaces the species normal flux with the catalytic wall flux.
    ub_cat_gam = ub_iso;
    ub_cat_gam_consistent = ub_iso;

    ub_all = [ub_in, ub_out, ub_iso, ub_slip, ub_grad, ub_noncat, ...
              ub_cat, ub_cat_gam, ub_cat_gam_consistent];

    if nargin < 14 || isempty(ib) || ib == 0
        ub = ub_all;
    elseif ib >= 1 && ib <= size(ub_all, 2)
        ub = ub_all(:, ib);
    else
        error('ubouaxialnd:InvalidBoundaryIndex', ...
            'Unsupported CNS5air axial boundary index ib = %d.', ib);
    end
end
