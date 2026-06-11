function s = sourcend(u, q, w, v, x, t, mu, eta)
    ns = 5;
    nd = length(x);

    % Nondimensional reference scales
    rho_scale   = eta(1);
    u_scale     = eta(2);
    T_scale     = eta(4);
    L_scale     = eta(8);
    omega_scale = rho_scale * u_scale / L_scale;

    rho_i_dim = u(1:ns) * rho_scale;
    T_dim = w(1) * T_scale;
    omega_i = kineticsource(T_dim, rho_i_dim);

    s = omega_i / omega_scale;
    s(ns+1:ns+nd+1) = 0;
end
