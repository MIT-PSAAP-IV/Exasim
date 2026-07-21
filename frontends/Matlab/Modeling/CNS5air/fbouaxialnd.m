function fb = fbouaxialnd(u, q, w, v, x, t, mu, eta, uh, n, tau, uinf, param, ib)
%FBOUAXIALND LDG boundary numerical flux for axial five-species reacting air.
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
    uh  = uh(:);
    w   = w(:);
    mu  = mu(:);
    eta = eta(:);
    n   = n(:);

    if numel(u) ~= ncu
        error('fbouaxialnd:InvalidUSize', 'u must have length %d, but got %d.', ncu, numel(u));
    end
    if numel(uh) ~= ncu
        error('fbouaxialnd:InvalidTraceSize', 'uh must have length %d, but got %d.', ncu, numel(uh));
    end
    if numel(w) < 1
        error('fbouaxialnd:InvalidWSize', 'w must contain at least one entry.');
    end
    if numel(mu) < 12
        error('fbouaxialnd:InvalidMuSize', 'mu must contain at least 12 entries.');
    end
    if numel(eta) < ncu + 2*ns
        error('fbouaxialnd:InvalidEtaSize', ...
            'eta must contain [ub(%d); Ycat(%d); gamma(%d)].', ncu, ns, ns);
    end

    irho = 1:ns;
    T_scale = mu(4);
    T_wall = mu(12);
    gam_i = eta(ncu + ns + (1:ns));

    qmat = cns5air_axial_qmat(q, ncu, nd);
    ub = ubouaxialnd(u, q, w, v, x, t, mu, eta, uh, n, tau, uinf, param, 0);

    fb_all = 0*ub;
    w_wall = w;
    w_wall(1) = T_wall / T_scale;

    for k = 1:size(ub, 2)
        wk = w;
        if k == 3 || k == 6 || k == 7 || k == 8 || k == 9
            wk = w_wall;
        end
        fb_all(:, k) = cns5air_axial_normal_flux(ub(:, k), q, wk, v, x, t, mu, eta, n) ...
                     + tau .* (ub(:, k) - uh);
    end

    % 5) Zero-gradient condition follows fbouhdgaxialnd.m: prescribe q·n
    % directly instead of evaluating a physical flux from an exterior state.
    fb_all(:, 5) = qmat * n + tau .* (u - uh);

    % 8-9) Catalytic wall species fluxes are flux boundary conditions. Use
    % the isothermal wall state for the axial viscous species flux, then
    % replace only the species block.
    u_iso = ub(:, 3);
    [fn_species_iso, rho_i_wall_dim, Mw, RU] = cns5air_axial_wall_species_flux( ...
        u_iso, q, w_wall, v, x, t, mu, eta, n);

    rho_scale = mu(1);
    u_scale = mu(2);

    mdot_cat = 0*fn_species_iso;
    for is = 1:ns
        mdot_cat(is) = gam_i(is) ...
            * sqrt(T_wall / (2*pi)) ...
            * sqrt(RU / Mw(is)) ...
            * rho_i_wall_dim(is);
    end
    fb_all(:, 8) = fb_all(:, 3);
    fb_all(irho, 8) = fn_species_iso(1:ns) - mdot_cat / (rho_scale * u_scale) ...
                    + tau .* (u(irho) - uh(irho));

    iN  = 1;
    iO  = 2;
    iN2 = 4;
    iO2 = 5;
    mdotN_sink = gam_i(iN) * sqrt(T_wall / (2*pi)) ...
        * sqrt(RU / Mw(iN)) * rho_i_wall_dim(iN);
    mdotO_sink = gam_i(iO) * sqrt(T_wall / (2*pi)) ...
        * sqrt(RU / Mw(iO)) * rho_i_wall_dim(iO);

    mdot_wall_dim = 0*fn_species_iso;
    mdot_wall_dim(iN)  = -mdotN_sink;
    mdot_wall_dim(iO)  = -mdotO_sink;
    mdot_wall_dim(iN2) =  mdotN_sink;
    mdot_wall_dim(iO2) =  mdotO_sink;

    fb_all(:, 9) = fb_all(:, 3);
    fb_all(irho, 9) = fn_species_iso(1:ns) + mdot_wall_dim / (rho_scale * u_scale) ...
                    + tau .* (u(irho) - uh(irho));

    if nargin < 14 || isempty(ib) || ib == 0
        fb = fb_all;
    elseif ib >= 1 && ib <= size(fb_all, 2)
        fb = fb_all(:, ib);
    else
        error('fbouaxialnd:InvalidBoundaryIndex', ...
            'Unsupported CNS5air axial boundary index ib = %d.', ib);
    end
end

function qmat = cns5air_axial_qmat(q, ncu, nd)
    if ismatrix(q) && ~isvector(q)
        qmat = q;
    else
        qmat = reshape(q(:), [ncu, nd]);
    end
end

function fn = cns5air_axial_normal_flux(u, q, w, v, x, t, mu, eta, n)
    nd = numel(x);
    if nd == 1
        f = fluxaxial1d(u, q, w, v, x, t, mu, eta);
        fn = f(:,1) * n(1);
    elseif nd == 2
        f = fluxaxial2d(u, q, w, v, x, t, mu, eta);
        fn = f(:,1) * n(1) + f(:,2) * n(2);
    elseif nd == 3
        f = fluxaxial3d(u, q, w, v, x, t, mu, eta);
        fn = f(:,1) * n(1) + f(:,2) * n(2) + f(:,3) * n(3);
    else
        error('fbouaxialnd:InvalidDimension', 'Unsupported spatial dimension nd = %d.', nd);
    end
end

function [fn_species_iso, rho_i_wall_dim, Mw, RU] = cns5air_axial_wall_species_flux(u, q, w_wall, v, x, t, mu, eta, n)
    ns = 5;
    nd = numel(x);
    rho_scale = mu(1);

    [~, Mw, RU] = thermodynamicsModels();
    Mw = Mw(:);
    rho_i_wall_dim = u(1:ns) * rho_scale;

    if nd == 1
        [~, f_species_iso] = fluxaxial1d(u, q, w_wall, v, x, t, mu, eta);
        fn_species_iso = f_species_iso(:,1) * n(1);
    elseif nd == 2
        [~, ~, ~, f_species_iso] = fluxaxial2d(u, q, w_wall, v, x, t, mu, eta);
        fn_species_iso = f_species_iso(:,1) * n(1) + f_species_iso(:,2) * n(2);
    elseif nd == 3
        [~, ~, ~, ~, f_species_iso] = fluxaxial3d(u, q, w_wall, v, x, t, mu, eta);
        fn_species_iso = f_species_iso(:,1) * n(1) + f_species_iso(:,2) * n(2) ...
                       + f_species_iso(:,3) * n(3);
    else
        error('fbouaxialnd:InvalidDimension', 'Unsupported spatial dimension nd = %d.', nd);
    end
end
