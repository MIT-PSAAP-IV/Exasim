function s = source2d(u, q, w, v, x, t, mu, eta)
    
    ns = 5;
    rmin = 0.0;

    [species_thermo_structs, Mw, ~] = thermodynamicsModels();
    kinetics_params = kinetics();

    % Nondimensional params
    rho_scale   = eta(1);
    u_scale     = eta(2);
    rhoe_scale  = eta(3);
    T_scale     = eta(4);
    mu_scale    = eta(5);
    kappa_scale = eta(6);
    cp_scale    = eta(7);
    L_scale     = eta(8);
    omega_scale = rho_scale * u_scale / L_scale;
    alphaClip = 1e12;

    % Mutation outputs
    rho_i_dim = u(1:ns) * rho_scale;
    % rho_i_dim = 0 + lmax(rho_i_dim,alphaClip); %subspecies density
    T = w(1) * T_scale;
    omega_i = netProductionRatesTotal(rho_i_dim, T, Mw, kinetics_params, species_thermo_structs);

    s(1:ns) = omega_i / omega_scale;
    s(ns+1) = 0.0;
    s(ns+2) = 0.0;
    s(ns+3) = 0.0;
end
