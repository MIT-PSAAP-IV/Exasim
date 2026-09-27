function pde = pdemodel
    pde.mass = @mass;
    pde.flux = @flux;
    pde.source = @source;
    pde.fbouhdg = @fbouhdg;
    pde.fbou = @fbou;
    pde.ubou = @ubou;
    pde.initu = @initu;
    pde.avfield = @avfield;
    pde.visscalars = @visscalars;
    pde.sourcew = @sourcew;
    pde.initw = @initw;
    pde.eos = @eos;
end

function m = mass(u, q, w, v, x, t, mu, eta)
    ns = 5;
    ndim = numel(x);
    m = sym(ones(ns + ndim + 1, 1));
end

function f = flux(u, q, w, v, x, t, mu, eta)
    v = artificialviscosity(v, mu);
    ndim = numel(x);
    if ndim==1
      f = fluxcart1d(u, q, w, v, x, t, mu, eta);
    elseif ndim==2
      f = fluxcart2d(u, q, w, v, x, t, mu, eta);
    elseif ndim==3
      f = fluxcart3d(u, q, w, v, x, t, mu, eta);
    end
end

function s = source(u, q, w, v, x, t, mu, eta)    
    s = sourcend(u, q, w, v, x, t, mu, eta);    
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    ub = 0*u;
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    fb = 0*u;
end

function u0 = initu(x, mu, eta)
    ns = 5;
    ndim = numel(x);
    u0 = sym(ones(ns + ndim + 1, 1));    
end

function w0 = initw(x, mu, eta)
    w0 = sym(ones(1,1));
end

function f = eos(u, q, w, v, x, t, mu, eta)
    f = eosnd(u, q, w, v, x, t, mu, eta);    
end

function f = sourcew(u, q, w, v, x, t, mu, eta)
    f = eosnd(u, q, w, v, x, t, mu, eta);        
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
  fb = fbouhdgnd(u, q, w, artificialviscosity(v, mu), x, t, mu, eta, uhat, n, tau);
end

function f = avfield(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
    ns = 5;
    nch = ns + numel(x) + 1;
    rho = sum(u(1:ns));
    ux = u(ns+1)/rho;
    uy = u(ns+2)/rho;
    rhox = sum(q(1:ns));
    rhoy = sum(q(nch+(1:ns)));
    compression = (q(ns+1) - rhox*ux)/rho ...
                + (q(nch+ns+2) - rhoy*uy)/rho;
    f = limiting(compression*tanh(mu(end-2)*v(1)), ...
                 0, mu(end-3), 1.0e3, 0);
end

function s = visscalars(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
    [~, Mw, ~] = thermodynamicsModels();
    s = pressure(mu(4)*w(1), mu(1)*u(1:5), Mw)/mu(3);
end

function va = artificialviscosity(v, mu)
    av = (mu(end-1) + mu(end)*v(2))*tanh(mu(end-2)*v(1));
    va = [av; av];
end
