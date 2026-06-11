function f = eosnd(u, q, w, v, x, t, mu, eta)    
    ns = 5;
    nd = length(x);

    % Nondimensional reference scales
    rho_scale   = mu(1);
    u_scale     = mu(2);
    rhoe_scale  = mu(3);
    T_scale     = mu(4);
    
    for i = 1:ns
      u(i) = lmax(u(i), 1e6);
    end
    rho_i = u(1:ns) * rho_scale;
    if nd == 1 
      rhou  = u(ns+1) * (rho_scale * u_scale);
      rhoE  = u(ns+2) * rhoe_scale;
      rhoe = (rhoE - 0.5 * (rhou * rhou) / sum(rho_i));      
    elseif nd == 2
      rhou  = u(ns+1) * (rho_scale * u_scale);
      rhov  = u(ns+2) * (rho_scale * u_scale);
      rhoE  = u(ns+3) * rhoe_scale;
      rhoe = (rhoE - 0.5 * (rhou * rhou + rhov * rhov) / sum(rho_i));
    elseif nd == 3
      rhou  = u(ns+1) * (rho_scale * u_scale);
      rhov  = u(ns+2) * (rho_scale * u_scale);
      rhow  = u(ns+3) * (rho_scale * u_scale);
      rhoE  = u(ns+4) * rhoe_scale;
      rhoe = (rhoE - 0.5 * (rhou * rhou + rhov * rhov + rhow * rhow) / sum(rho_i));  
    end
    f = equationofstate(w(1) * T_scale, rho_i, rhoe);
end
