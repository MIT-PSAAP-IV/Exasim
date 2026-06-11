function f = fluxcart2d(u, q, w, v, x, t, mu, eta)    
    ns = 5;
    nch = ns + 3;

    % Molecular weights are required for the equation of state.
    [~, Mw, ~] = thermodynamicsModels();

    % Nondimensional reference scales / parameters
    rho_scale   = eta(1);
    u_scale     = eta(2);
    rhoe_scale  = eta(3);
    T_scale     = eta(4);
    mu_scale    = eta(5);
    kappa_scale = eta(6);
    L_scale     = eta(8);
    Ec          = eta(9);
    Pr          = eta(10);
    Re          = eta(11);

    % Conservative variables
    rho_i = zeros(ns,1);
    rho = 0;
    for ispecies = 1:ns
        rho_i(ispecies) = u(ispecies);
        rho = rho + rho_i(ispecies);
    end

    rhou = u(ns+1);
    rhov = u(ns+2);
    rhoE = u(ns+3);

    % NOTE:
    % The current Exasim first-order formulation used here assumes q = -grad(u).
    % If your formulation instead uses q = grad(u), remove the minus signs below.
    drho_dx_i = -q(1:ns);
    drhou_dx  = -q(ns+1);
    drhov_dx  = -q(ns+2);
    drhoE_dx  = -q(ns+3);

    drho_dy_i = -q((nch+1):(nch+ns));
    drhou_dy  = -q(nch+ns+1);
    drhov_dy  = -q(nch+ns+2);
    drhoE_dy  = -q(nch+ns+3);

    av = v(1);

    rhoinv = 1.0 / rho;
    uv = rhou * rhoinv;
    vv = rhov * rhoinv;
    E = rhoE * rhoinv;

    % Dimensional thermodynamic state
    rho_i_dim = rho_i * rho_scale;
    T = w(1);
    T_dim = T * T_scale;

    % Pressure scaling: here p_ref = rhoe_scale.
    p_dim = pressure(T_dim, rho_i_dim, Mw);
    p = p_dim / rhoe_scale;
    H = E + p * rhoinv;

    % Preallocate fluxes
    fi = zeros(nch,2);
    fv = zeros(nch,2);

    % Inviscid + artificial viscosity fluxes
    for ispecies = 1:ns
        fi(ispecies,1) = rho_i(ispecies) * uv - av * drho_dx_i(ispecies);
        fi(ispecies,2) = rho_i(ispecies) * vv - av * drho_dy_i(ispecies);
    end
    fi(ns+1,1) = rhou * uv + p - av * drhou_dx;
    fi(ns+2,1) = rhov * uv     - av * drhov_dx;
    fi(ns+3,1) = rhou * H      - av * drhoE_dx;

    fi(ns+1,2) = rhou * vv     - av * drhou_dy;
    fi(ns+2,2) = rhov * vv + p - av * drhov_dy;
    fi(ns+3,2) = rhov * H      - av * drhoE_dy;

    % Transport properties and temperature sensitivities
    [dT_drho_i_dim, dT_drhoe_dim, D_vec, h_vec, mu_d_dim, kappa_dim] = ...
        transportcoefficients(T_dim, rho_i_dim);

    drho_dx = sum(drho_dx_i);
    drho_dy = sum(drho_dy_i);

    du_dx = (drhou_dx - drho_dx * uv) * rhoinv;
    dv_dx = (drhov_dx - drho_dx * vv) * rhoinv;
    du_dy = (drhou_dy - drho_dy * uv) * rhoinv;
    dv_dy = (drhov_dy - drho_dy * vv) * rhoinv;

    uTu2 = 0.5 * (uv * uv + vv * vv);
    duTu2_dx = uv * du_dx + vv * dv_dx;
    duTu2_dy = uv * du_dy + vv * dv_dy;

    % dT/drho_i and dT/drhoe in dimensionless variables
    dT_drho_i = dT_drho_i_dim * rho_scale / T_scale;
    dT_drhoe  = dT_drhoe_dim  * rhoe_scale / T_scale;

    dre_drho  = -Ec * uTu2;
    dre_duTu2 = -Ec * rho;
    dre_drhoE =  Ec;

    dre_dx = dre_drho * drho_dx + dre_duTu2 * duTu2_dx + dre_drhoE * drhoE_dx;
    dre_dy = dre_drho * drho_dy + dre_duTu2 * duTu2_dy + dre_drhoE * drhoE_dy;

    dT_dx = sum(dT_drho_i .* drho_dx_i) + dT_drhoe * dre_dx;
    dT_dy = sum(dT_drho_i .* drho_dy_i) + dT_drhoe * dre_dy;

    % Nondimensionalize transport properties
    h_scale = u_scale^2;
    D_scale = u_scale * L_scale;   % FIX: diffusion coefficient has units L^2 / t

    mu_d  = mu_d_dim / mu_scale;
    kappa = kappa_dim / kappa_scale;
    D_vec = D_vec / D_scale;
    h_vec = h_vec / h_scale;

    % Species mass fractions and mixture-averaged diffusion fluxes
    dY_dx_i = (drho_dx_i * rho - rho_i * drho_dx) * rhoinv * rhoinv;
    dY_dy_i = (drho_dy_i * rho - rho_i * drho_dy) * rhoinv * rhoinv;

    J_i_x = -rho * D_vec .* dY_dx_i + rho_i .* sum(D_vec .* dY_dx_i);
    J_i_y = -rho * D_vec .* dY_dy_i + rho_i .* sum(D_vec .* dY_dy_i);

    % Newtonian stress tensor
    beta = 0; % nondimensional bulk viscosity; currently disabled
    divu = du_dx + dv_dy;
    txx = mu_d * (2.0/3.0) * (2 * du_dx - dv_dy) / Re + beta * divu / Re;
    txy = mu_d * (du_dy + dv_dx) / Re;
    tyy = mu_d * (2.0/3.0) * (2 * dv_dy - du_dx) / Re + beta * divu / Re;

    % Viscous fluxes
    for i = 1:ns
        fv(i,1) = -J_i_x(i);
        fv(i,2) = -J_i_y(i);
    end

    fv(ns+1,1) = txx;
    fv(ns+2,1) = txy;
    fv(ns+3,1) = uv * txx + vv * txy - (sum(h_vec .* J_i_x) - kappa * dT_dx / (Re * Pr * Ec));

    fv(ns+1,2) = txy;
    fv(ns+2,2) = tyy;
    fv(ns+3,2) = uv * txy + vv * tyy - (sum(h_vec .* J_i_y) - kappa * dT_dy / (Re * Pr * Ec));

    f = fi - fv;
end
