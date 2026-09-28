function f = eosnd(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
    ns = 5;
    nd = length(x);

    % Nondimensional reference scales
    rho_scale   = mu(1);
    u_scale     = mu(2);
    rhoe_scale  = mu(3);
    T_scale     = mu(4);

    rhoFloor = 1e-12;
    alphaRho = 1e6;
    rho_nd = 0*u(1:ns);
    for i = 1:ns
      rho_nd(i) = rhoFloor + lmax(u(i)-rhoFloor,alphaRho);
    end
    rho_i = rho_scale*rho_nd;
    rho = sum(rho_i);

    momentum2 = 0*u(1);
    for d = 1:nd
      momentum = u(ns+d)*(rho_scale*u_scale);
      momentum2 = momentum2 + momentum*momentum;
    end
    rhoE = u(ns+nd+1)*rhoe_scale;
    rhoe = rhoE-0.5*momentum2/rho;

    T = w(1)*T_scale;
    %f = equationofstate(T,rho_i,rhoe);
    f = equationofstate_scaled(T,rho_i,rhoe,rhoe_scale);
end
