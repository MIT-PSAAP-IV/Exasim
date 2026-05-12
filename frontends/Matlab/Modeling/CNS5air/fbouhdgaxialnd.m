function fb = fbouhdgaxialnd(u, q, w, v, x, t, mu, eta, uhat, n, tau)
%FBOUHDGND HDG boundary terms for reacting-air HDG model.
%
% State ordering (ns = 5 species):
%   u = [rho_1, ..., rho_ns, rho*u_1, ..., rho*u_nd, rhoE]^T
%
% Inputs
%   u     : interior state, ncu x 1
%   q     : gradient-like variable, either (ncu*nd) x 1 or ncu x nd
%   w     : auxiliary state; assumed w(1) = nondimensional temperature
%   x     : spatial coordinate, nd x 1
%   mu    : parameter vector
%           mu(1)  = rho_scale
%           mu(2)  = u_scale
%           mu(3)  = rhoe_scale
%           mu(4)  = T_scale
%           mu(12) = T_wall [K]
%   eta   : boundary-data vector
%           eta(1:ncu)                 = prescribed inflow state ub
%           eta(ncu+1:ncu+ns)          = supercatalytic wall mass fractions
%           eta(ncu+ns+1:ncu+2*ns)     = partial-catalysis coefficients gamma_i
%   uhat  : trace state, ncu x 1
%   n     : outward unit normal, nd x 1
%   tau   : stabilization parameter (scalar or ncu x 1 compatible)
%
% Output
%   fb    : ncu x 8 matrix whose columns correspond to:
%           [inflow, outflow, isothermal wall, slip wall, zero-gradient,
%            noncatalytic wall, supercatalytic wall, partially catalytic wall]
%
% Notes
%   - This implementation assumes ns = 5 reacting-air species.
%   - Supports nd = 2 or 3.

    % -------------------------------------------------
    % Basic dimensions
    % -------------------------------------------------
    ns = 5;
    nd = numel(x);
    ncu = ns + nd + 1;

    % -------------------------------------------------
    % Force column vectors
    % -------------------------------------------------
    u    = u(:);
    uhat = uhat(:);
    x    = x(:);
    n    = n(:);
    w    = w(:);
    mu   = mu(:);
    eta  = eta(:);

    % -------------------------------------------------
    % Validate sizes
    % -------------------------------------------------
    if numel(u) ~= ncu
        error('fbouhdgnd:InvalidUSize', ...
            'u must have length %d, but got %d.', ncu, numel(u));
    end

    if numel(uhat) ~= ncu
        error('fbouhdgnd:InvalidUhatSize', ...
            'uhat must have length %d, but got %d.', ncu, numel(uhat));
    end

    if numel(n) ~= nd
        error('fbouhdgnd:InvalidNormalSize', ...
            'n must have length nd = %d, but got %d.', nd, numel(n));
    end

    if numel(mu) < 12
        error('fbouhdgnd:InvalidMuSize', ...
            'mu must contain at least 12 entries.');
    end

    if numel(w) < 1
        error('fbouhdgnd:InvalidWSize', ...
            'w must contain at least one entry (temperature).');
    end

    % eta = [ub ; Ycat ; gamma]
    n_eta_min = ncu + 2*ns;
    if numel(eta) < n_eta_min
        error('fbouhdgnd:InvalidEtaSize', ...
            'eta must contain at least %d entries: [ub(%d); Ycat(%d); gamma(%d)].', ...
            n_eta_min, ncu, ns, ns);
    end

    % -------------------------------------------------
    % q handling
    % -------------------------------------------------
    if ismatrix(q) && ~isvector(q)
        [rq, cq] = size(q);
        if rq ~= ncu || cq ~= nd
            error('fbouhdgnd:InvalidQMatrixSize', ...
                'If q is a matrix, it must have size [%d, %d]. Got [%d, %d].', ...
                ncu, nd, rq, cq);
        end
        qmat = q;
    else
        q = q(:);
        if numel(q) ~= ncu * nd
            error('fbouhdgnd:InvalidQSize', ...
                'q must have size ncu*nd = %d, but got %d entries.', ...
                ncu * nd, numel(q));
        end
        qmat = reshape(q, [ncu, nd]);
    end

    % -------------------------------------------------
    % Indices
    % -------------------------------------------------
    irho  = 1:ns;
    irhou = ns + (1:nd);
    irhoE = ns + nd + 1;

    % -------------------------------------------------
    % Parameters
    % -------------------------------------------------
    rho_scale  = mu(1);
    u_scale    = mu(2);
    rhoe_scale = mu(3);
    T_scale    = mu(4);
    T_wall     = mu(12);

    ub      = eta(1:ncu);
    Y_i_cat = eta(ncu + (1:ns));
    gam_i   = eta(ncu + ns + (1:ns));

    % -------------------------------------------------
    % Thermodynamic data
    % -------------------------------------------------
    [~, Mw, RU] = thermodynamicsModels();
    Mw = Mw(:);

    % -------------------------------------------------
    % Common wall helpers
    % -------------------------------------------------
    v_wall = 0*n(:);
    
    % -------------------------------------------------
    % 1) Prescribed inflow / Dirichlet-like state
    % -------------------------------------------------
    f_in = tau .* (ub - uhat);

    % -------------------------------------------------
    % 2) Outflow / trace matching
    % -------------------------------------------------
    f_out = tau .* (u - uhat);

    % -------------------------------------------------
    % 3) Isothermal no-slip wall
    %    Keep species densities from interior state, set wall velocity to zero,
    %    and recompute rhoE using T_wall.
    % -------------------------------------------------
    rho_i_wall_dim = u(irho) * rho_scale;

    [rhoE_dim_wall, ~, ~] = energyFromSpecies(rho_i_wall_dim, T_wall, v_wall);
    rhoE_wall = rhoE_dim_wall / rhoe_scale;

    uwall = u;
    uwall(irhou) = 0;
    uwall(irhoE) = rhoE_wall;

    f_iso = tau .* (uwall - uhat);

    % -------------------------------------------------
    % 4) Slip wall
    %    Remove only the normal component of momentum.
    % -------------------------------------------------
    m  = u(irhou);
    mn = m.' * n;

    uslip = u;
    uslip(irhou) = m - mn .* n;

    f_slip = tau .* (uslip - uhat);

    % -------------------------------------------------
    % 5) Zero-gradient / Neumann-like condition
    % -------------------------------------------------
    qn = qmat * n;
    f_grad = qn + tau .* (u - uhat);

    % -------------------------------------------------
    % Flow thermodynamic state used in catalytic / noncatalytic walls
    % -------------------------------------------------
    T_flow     = w(1) * T_scale;
    rho_i_flow = u(irho) * rho_scale;
    rho_flow   = sum(rho_i_flow);
    p_flow = pressure(T_flow, rho_i_flow, Mw);

    % -------------------------------------------------
    % 6) Noncatalytic wall
    %    Keep wall mass fractions equal to flow mass fractions:
    %      rho_i_wall = (rho_wall / rho_flow) * rho_i_flow
    % -------------------------------------------------
    Smix = sum(rho_i_flow ./ Mw) / rho_flow;
    rho_noncat = p_flow / (RU * T_wall * Smix);
    rho_i_noncat = (rho_noncat / rho_flow) * rho_i_flow;
    [rhoE_dim_noncat, ~, ~] = energyFromSpecies(rho_i_noncat, T_wall, v_wall);

    u_noncat = 0*uhat;
    u_noncat(irho)  = rho_i_noncat / rho_scale;
    u_noncat(irhou) = 0;
    u_noncat(irhoE) = rhoE_dim_noncat / rhoe_scale;
    f_noncat = tau .* (u_noncat - uhat);

    % -------------------------------------------------
    % 7) Supercatalytic wall
    %    Prescribe wall mass fractions Y_i_cat at T_wall and p_flow.
    % -------------------------------------------------
    Y_i_cat = Y_i_cat(:);
    Ysum = sum(Y_i_cat);
    % Normalize if slightly inconsistent
    Y_i_cat = Y_i_cat / Ysum;

    rho_cat   = densityFromMassFractions(T_wall, p_flow, Y_i_cat, Mw);
    rho_i_cat = rho_cat * Y_i_cat;
    [rhoE_dim_cat, ~, ~] = energyFromSpecies(rho_i_cat, T_wall, v_wall);

    u_cat = 0*uhat;
    u_cat(irho)  = rho_i_cat / rho_scale;
    u_cat(irhou) = 0;
    u_cat(irhoE) = rhoE_dim_cat / rhoe_scale;
    f_cat = tau .* (u_cat - uhat);

    % -------------------------------------------------
    % 8) Partially catalytic wall
    %    Modify the normal species viscous flux with a finite catalytic flux.
    % -------------------------------------------------
    gam_i = gam_i(:);

    % Use the same isothermal/no-slip state used in f_iso
    uf = u;
    uf(irhou) = 0;
    uf(irhoE) = rhoE_wall;
    w_wall = T_wall / T_scale;

    %f_species_iso = flux_visc_species(uf, q, w_wall, v, x, t, mu, eta);
    if nd == 1
      [~, f_species_iso] = fluxaxial1d(uf, q, w_wall, v, x, t, mu, eta);
    elseif nd == 2      
      [~, ~, ~, f_species_iso] = fluxaxial2d(uf, q, w_wall, v, x, t, mu, eta);
    elseif nd == 3      
      [~, ~, ~, ~, f_species_iso] = fluxaxial3d(uf, q, w_wall, v, x, t, mu, eta);
    end

    % Expect species flux tensor with at least ns rows and nd columns
    if size(f_species_iso, 1) < ns || size(f_species_iso, 2) < nd
        error('fbouhdgnd:InvalidSpeciesFluxSize', ...
            'flux_visc_species must return at least an ns-by-nd array.');
    end

    % Normal viscous species flux
    fn_species_iso = f_species_iso(:,1) * 0;
    for d = 1:nd
        fn_species_iso = fn_species_iso + f_species_iso(:,d) * n(d);
    end

    % Catalytic wall mass flux (dimensional)
    mdot_cat = 0*fn_species_iso;
    for is = 1:ns
        mdot_cat(is) = gam_i(is) ...
                  * sqrt(T_wall / (2*pi)) ...
                  * sqrt(RU / Mw(is)) ...
                  * u(is) * rho_scale;
    end

    % Add catalytic contribution in nondimensional form
    fn_species_iso = fn_species_iso(1:ns) - mdot_cat / (rho_scale * u_scale);

    % Start from isothermal wall BC and replace species block
    f_cat_gam = f_iso;
    f_cat_gam(irho) = fn_species_iso + tau .* (u(irho) - uhat(irho));

    % -------------------------------------------------
    % 8) Partially catalytic wall
    %    Stoichiometrically consistent recombination model:
    %       2N -> N2
    %       2O -> O2
    %
    %    Species ordering:
    %       1:N, 2:O, 3:NO, 4:N2, 5:O2
    %
    %    Here fluxcart* returns the viscous species flux J_i,
    %    so the normal flux is J_i·n.
    % -------------------------------------------------

    % Dimensional species densities near the wall
    rho_i_wall_dim = uf(irho) * rho_scale;

    % Indices
    iN  = 1;
    iO  = 2;
    iN2 = 4;
    iO2 = 5;

    % Positive sink magnitudes [kg/(m^2 s)] for atoms removed from gas
    mdotN_sink = gam_i(iN) * sqrt(T_wall / (2*pi)) ...
               * sqrt(RU / Mw(iN)) * rho_i_wall_dim(iN);

    mdotO_sink = gam_i(iO) * sqrt(T_wall / (2*pi)) ...
               * sqrt(RU / Mw(iO)) * rho_i_wall_dim(iO);

    % Stoichiometrically consistent gas-side wall mass fluxes:
    % atomic species removed, molecular species injected
    mdot_wall_dim = 0*fn_species_iso;
    mdot_wall_dim(iN)  = -mdotN_sink;
    mdot_wall_dim(iO)  = -mdotO_sink;
    mdot_wall_dim(iN2) = +mdotN_sink;
    mdot_wall_dim(iO2) = +mdotO_sink;

    % Since fn_species_iso stores J_i·n, directly add mdot_wall_dim/scale
    fn_species_cat = fn_species_iso(1:ns) + mdot_wall_dim / (rho_scale * u_scale);

    % Start from isothermal wall BC and replace species block
    f_cat_gam_consistent = f_iso;
    f_cat_gam_consistent(irho) = fn_species_cat + tau .* (u(irho) - uhat(irho));
    
    % -------------------------------------------------
    % Collect all boundary conditions
    % -------------------------------------------------
    fb = [f_in, f_out, f_iso, f_slip, f_grad, f_noncat, f_cat, f_cat_gam, f_cat_gam_consistent];
end
