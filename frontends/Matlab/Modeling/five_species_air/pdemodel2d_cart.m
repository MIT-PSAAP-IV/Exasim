function pde = pdemodel
    pde.flux = @flux;
    pde.source = @source;
    pde.eos = @eos;
end

function f = flux(u, q, w, v, x, t, mu, eta)
    nch = 8;
    ns = 5;

    % ndim = 2;
    [species_thermo_structs, Mw, ~] = thermodynamicsModels();

    % Nondimensional params
    rho_scale   = eta(1);
    u_scale     = eta(2);
    rhoe_scale  = eta(3);
    T_scale     = eta(4);
    mu_scale    = eta(5);
    kappa_scale = eta(6);

    rho_i = sym(zeros(ns,1));
    rho = sym(0);

    % Conservative Variables
    for ispecies = 1:ns
        % rho_i(ispecies) = 0 + lmax(u(ispecies)-0,alphaClip); %subspecies density
        rho_i(ispecies) = u(ispecies);
        rho = rho + rho_i(ispecies); %total mixture density
    end

    rhou = u(ns+1);
    rhov = u(ns+2);
    rhoE = u(ns+3);

    drho_dx_i = -q(1:ns);
    drhou_dx  = -q(ns+1);
    drhov_dx  = -q(ns+2);
    drhoE_dx  = -q(ns+2+1);
    drho_dy_i = -q((nch+1:nch+ns));
    drhou_dy  = -q(nch+ns+1);
    drhov_dy  = -q(nch+ns+2);
    drhoE_dy  = -q(nch+ns+2+1);
    av = v(1);

    rhoinv = 1.0 / rho;
    uv = rhou * rhoinv; %velocity
    vv = rhov * rhoinv;
    E = rhoE * rhoinv; %energy

    % Mutation outputs
    rho_i_dim = rho_i * rho_scale;
    T = w(1);
    T_dim = T * T_scale;
    p_dim = pressure(T_dim, rho_i_dim, Mw);
    p = p_dim / rhoe_scale;

    H = E + p*rhoinv; %enthalpy

    % Inviscid + AV Fluxes
    for ispecies = 1:ns
        fi(ispecies,1) = rho_i(ispecies) * uv - av.*drho_dx_i(ispecies);
    end
    fi(ns + 1,1) = rhou * uv + p     - av.*drhou_dx;
    fi(ns + 2,1) = rhov * uv         - av.*drhov_dx;
    fi(ns + 3,1) = rhou * H          - av.*drhoE_dx;

    for ispecies = 1:ns
        fi(ispecies,2) = rho_i(ispecies) * vv - av.*drho_dy_i(ispecies);
    end
    fi(ns + 1,2) = rhou * vv      - av.*drhou_dy;
    fi(ns + 2,2) = rhov * vv + p  - av.*drhov_dy;
    fi(ns + 3,2) = rhov * H       - av.*drhoE_dy;

    % Viscous fluxes
    beta = 0;
    Ec          = eta(9);
    Pr          = eta(10);
    Re          = eta(11);

    [dT_drho_i_dim, dT_drhoe_dim, D_vec, h_vec, mu_d_dim, kappa_dim] = transportcoefficients(T_dim, rho_i_dim);

    drho_dx = sum(drho_dx_i);
    drho_dy = sum(drho_dy_i);
    %X = X_i(rho_i_dim,Mw);
    uv = rhou * rhoinv; %velocity
    vv = rhov * rhoinv;
    %E = rhoE .* rhoinv; 
    du_dx = (drhou_dx - drho_dx*uv)*rhoinv;
    dv_dx = (drhov_dx - drho_dx*vv)*rhoinv;
    du_dy = (drhou_dy - drho_dy*uv)*rhoinv;
    dv_dy = (drhov_dy - drho_dy*vv)*rhoinv;
    uTu2      = 0.5 * (uv * uv + vv * vv);
    duTu2_dx  = uv * du_dx + vv * dv_dx; 
    duTu2_dy  = uv * du_dy + vv * dv_dy;
    
    dT_drho_i = dT_drho_i_dim / T_scale * rho_scale;
    dT_drhoe = dT_drhoe_dim / T_scale * rhoe_scale;

    dre_drho  = Ec*-uTu2;
    dre_duTu2 = Ec*-rho;
    dre_drhoE = Ec*1.0;
    dre_dx    = dre_drho * drho_dx + dre_duTu2 * duTu2_dx + dre_drhoE * drhoE_dx;
    dre_dy    = dre_drho * drho_dy + dre_duTu2 * duTu2_dy + dre_drhoE * drhoE_dy;
    dT_dx     = sum(dT_drho_i .* drho_dx_i) +  dT_drhoe * dre_dx;
    dT_dy     = sum(dT_drho_i .* drho_dy_i) +  dT_drhoe * dre_dy;

    h_scale = u_scale^2;
    D_scale = u_scale;

    mu_d = mu_d_dim / mu_scale;
    kappa = kappa_dim / kappa_scale;
    D_vec = D_vec / D_scale;
    h_vec = h_vec / h_scale;

    %%%%%%%% Calculation of J_i
    dY_dx_i = (drho_dx_i * rho - rho_i * drho_dx) * rhoinv * rhoinv;
    dY_dy_i = (drho_dy_i * rho - rho_i * drho_dy) * rhoinv * rhoinv;

    J_i_x = -rho * D_vec .* dY_dx_i + rho_i .* sum(D_vec .* dY_dx_i);
    J_i_y = -rho * D_vec .* dY_dy_i + rho_i .* sum(D_vec .* dY_dy_i);

    %%%%%%%% Stress tensor tau
    txx = mu_d * 2.0/3.0 * (2 * du_dx - dv_dy) / Re + beta * (du_dx + dv_dy);
    txy = mu_d * (du_dy + dv_dx) / Re;
    tyy = mu_d * 2.0/3.0 * (2 * dv_dy - du_dx) / Re + beta * (du_dx + dv_dy);

    % VISCOUS FLUX
    
    for i = 1:ns
        fv(i,1) = -J_i_x(i); 
        fv(i,2) = -J_i_y(i);
    end

    fv(ns + 1, 1) = txx;
    fv(ns + 2, 1) = txy;
    fv(ns + 3,1) = uv * txx + vv * txy - (sum(h_vec.*J_i_x) - kappa*dT_dx / (Re*Pr*Ec));
    
    fv(ns + 1, 2) = txy;
    fv(ns + 2, 2) = tyy;
    fv(ns + 3,2) = uv * txy + vv * tyy - (sum(h_vec.*J_i_y) - kappa*dT_dy / (Re*Pr*Ec));

    f = fi - fv;
end

function s = source(u, q, w, v, x, t, mu, eta)
    
    ns = 5;
    % Nondimensional params
    rho_scale   = eta(1);
    u_scale     = eta(2);
    T_scale     = eta(4);
    L_scale     = eta(8);
    omega_scale = rho_scale * u_scale / L_scale;

    % Mutation outputs
    rho_i_dim = u(1:ns) * rho_scale;
    % rho_i_dim = 0 + lmax(rho_i_dim,alphaClip); %subspecies density
    T = w(1) * T_scale;
    omega_i = kineticsource(T, rho_i_dim);

    s(1:ns) = omega_i / omega_scale;
    s(ns+1) = 0.0;
    s(ns+2) = 0.0;
    s(ns+3) = 0.0;
end

function f = eos(u, q, w, v, x, t, mu, eta)
% Nondimensional params
    kinetics_params = kinetics();
    ns = kinetics_params.ns;
    rho_scale   = eta(1);
    u_scale     = eta(2);
    rhoe_scale  = eta(3);
    T_scale     = eta(4);
    Ec          = eta(9);
    
    rho_i = u(1:ns) * rho_scale;
    rhou = u(ns+1) * (rho_scale * u_scale);
    rhov = u(ns+2) * (rho_scale * u_scale);
    rhoE = u(ns+3) * rhoe_scale;

    rhoe = Ec * (rhoE - 0.5 * (rhou*rhou + rhov*rhov) / sum(rho_i));
    f = equationofstate(w(1)*T_scale, rho_i, rhoe);
end
