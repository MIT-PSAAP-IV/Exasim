function s = sourceaxialns_air5(u, q, w, v, x, t, mu, eta)
    nenergy = 1;
    ns = 5;
    nch = 8;
    % ndim = 2;
    [species_thermo_structs, Mw, RU] = thermodynamicsModels();

    % Nondimensional params
    rho_scale   = eta(1);
    u_scale     = eta(2);
    rhoe_scale  = eta(3);
    T_scale     = eta(4);
    mu_scale    = eta(5);
    kappa_scale = eta(6);
    cp_scale    = eta(7);
    L_scale     = eta(8);

    rho_i = sym(zeros(ns,1));
    rho = sym(0);

    % Conservative Variables
    for ispecies = 1:ns
    %         rho_i(ispecies) = rmin + lmax(u(ispecies)-rmin,alphaClip); %subspecies density
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

    for ispecies = 1:ns
        fi(ispecies,1) = rho_i(ispecies) * vv - av.*drho_dy_i(ispecies);
    end
    fi(ns + 1,1) = rhou * vv      - av.*drhou_dy;
    fi(ns + 2,1) = rhov * vv      - av.*drhov_dy;
    fi(ns + 3,1) = rhov * H       - av.*drhoE_dy;

    % Viscous fluxes
    beta = 0;
    Ec          = eta(9);
    Pr          = eta(10);
    Re          = eta(11);

    [blottner_structs, gupta_structs, gupta_mu_structs, gupta_kappa_structs] = transport();
    drho_dx = sum(drho_dx_i);
    drho_dy = sum(drho_dy_i);
    X = X_i(rho_i_dim,Mw);
    uv = rhou * rhoinv; %velocity
    vv = rhov * rhoinv;
    E = rhoE .* rhoinv; 
    du_dx = (drhou_dx - drho_dx*uv)*rhoinv;
    dv_dx = (drhov_dx - drho_dx*vv)*rhoinv;
    du_dy = (drhou_dy - drho_dy*uv)*rhoinv;
    dv_dy = (drhov_dy - drho_dy*vv)*rhoinv;
    uTu2      = 0.5 * (uv * uv + vv * vv);
    duTu2_dx  = uv * du_dx + vv * dv_dx; 
    duTu2_dy  = uv * du_dy + vv * dv_dy;

    Y = Y_i(rho_i_dim);
    denom = sum(rho_i_dim) * mixtureFrozenCvMass(T_dim, Mw, Y, species_thermo_structs);
    e_i = getEnergiesMass(T_dim, Mw, species_thermo_structs);
    dT_drho_i_dim = -e_i(:) ./ denom;
    dT_drhoe_dim = 1.0 / denom;

    dT_drho_i = dT_drho_i_dim / T_scale * rho_scale;
    dT_drhoe = dT_drhoe_dim / T_scale * rhoe_scale;

    dre_drho  = -uTu2;
    dre_duTu2 = -rho;
    dre_drhoE = 1.0;
    dre_dx    = dre_drho * drho_dx + dre_duTu2 * duTu2_dx + dre_drhoE * drhoE_dx;
    dre_dy    = dre_drho * drho_dy + dre_duTu2 * duTu2_dy + dre_drhoE * drhoE_dy;
    dT_dx     = sum(dT_drho_i .* drho_dx_i) +  dT_drhoe * dre_dx;
    dT_dy     = sum(dT_drho_i .* drho_dy_i) +  dT_drhoe * dre_dy;

    D_vec = averageDiffusionCoeffs(T_dim, X, Y, Mw, p_dim, gupta_structs);
    h_vec = getEnthalpiesMass(T_dim, Mw, species_thermo_structs);

    mu_i = speciesViscosities(T_dim, blottner_structs);
    % lambda_i = speciesConductivities(T, gupta_kappa_structs);
    phi_i = euckenPhi(mu_i, Mw, X);
    mu_d_dim = wilkeMixture(mu_i, X, phi_i);    
    lambda_i = mu_i .* (getCpsMass(T_dim, Mw, species_thermo_structs) + 5/4 * RU./Mw);
    kappa_dim = wilkeMixture(lambda_i, X, phi_i);

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
    y = x(2);
    % txx = mu_d * 2.0/3.0 * (2 * du_dx - dv_dy +  vv/y) / Re + beta * (du_dx + dv_dy);
    % txy = mu_d * (du_dy + dv_dx) / Re;
    % tyy = mu_d * 2.0/3.0 * (2 * dv_dy - du_dx +  vv/y) / Re + beta * (du_dx + dv_dy);
    txy = mu_d * (du_dy + dv_dx) / Re;
    tyy = mu_d * 2.0/3.0 * (2 * dv_dy - du_dx - vv / y) / Re;
    ttt = mu_d * 2.0/3.0 * (2 * vv / y - du_dx - dv_dy) / Re;
    % VISCOUS FLUX

    for i = 1:ns
        fv(i,1) = -J_i_y(i);
    end

    %[txy tyy ttt]

    fv(ns + 1, 1) = txy;
    fv(ns + 2, 1) = tyy - ttt;
    fv(ns + 3, 1) = uv * txy + vv * tyy - (sum(h_vec.*J_i_y) - kappa*dT_dy / (Re*Pr*Ec));
        
    ftot = fi - fv;
    s = -ftot(:) / y;

    ns = 5;
    kinetics_params = kinetics();

    % Nondimensional params
    omega_scale = rho_scale * u_scale / L_scale;

    % Mutation outputs
    rho_i_dim = u(1:ns) * rho_scale;
    T = w(1) * T_scale;
    omega_i = netProductionRatesTotal(rho_i_dim, T, Mw, kinetics_params, species_thermo_structs);
    
    s(1:ns) = s(1:ns) + omega_i / omega_scale;    
end

