function f = sourcew2d(u, q, w, v, x, t, mu, eta)
% Nondimensional params
    kinetics_params = kinetics();
    [species_thermo_structs, Mw, ~] = thermodynamicsModels();
    ns = kinetics_params.ns;
    rho_scale   = eta(1);
    u_scale     = eta(2);
    rhoe_scale  = eta(3);
    T_scale     = eta(4);
    mu_scale    = eta(5);
    kappa_scale = eta(6);
    cp_scale    = eta(7);
    L_scale     = eta(8);
    Ec          = eta(9);
    

    rho_i = u(1:ns) * rho_scale;
    rhou = u(ns+1) * (rho_scale * u_scale);
    rhov = u(ns+2) * (rho_scale * u_scale);
    rhoE = u(ns+3) * rhoe_scale;

    rhoe = (rhoE - 0.5 * (rhou*rhou + rhov*rhov) / sum(rho_i));
    rho_tilde = rho_i ./ Mw;
    alpha = -sum(rho_tilde);
    f = f_T(w(1)*T_scale, rho_tilde, rhoe, alpha, species_thermo_structs);
end
