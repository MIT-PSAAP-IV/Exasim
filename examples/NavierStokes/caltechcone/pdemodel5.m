function pde = pdemodel5
    pde.mass = @mass;
    pde.flux = @flux;
    pde.source = @source;
    pde.fbouhdg = @fbouhdg;
    pde.fbou = @fbou;
    pde.ubou = @ubou;
    pde.initu = @initu;
    pde.avfield = @avfield;
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
    vflux = v;   
    dynamic = v(end)*(1-v(end-2)) * v(end-3);
    fixed = v(end-1)*(v(end-2));
    vflux(end) = dynamic + fixed;

    f = fluxaxial2d(u, q, w, vflux, x, t, mu, eta);
end

function myavfield = avfield(u, q, w, v, x, t, mu, eta)
    gam = mu(1); %not used by sensor, can be whatever
    avcoeff = mu(13);
    myavfield = getavfield2dchem(x(2), u, q, v(9), gam, avcoeff, 2);
end



function s = source(u, q, w, v, x, t, mu, eta)
    vflux = v;   
    dynamic = v(end)*(1-v(end-2)) * v(end-3);
    fixed = v(end-1)*(v(end-2));
    vflux(end) = dynamic + fixed;
    v = vflux;
    s = sourceaxial2d(u, q, w, vflux, x, t, mu, eta);    
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    vflux = v;   
    dynamic = v(end)*(1-v(end-2)) * v(end-3);
    fixed = v(end-1)*(v(end-2));
    vflux(end) = dynamic + fixed;
    v = vflux;
    ub = ubouaxialnd(u, q, w, v, x, t, mu, eta, uhat, n, tau, eta, mu, 0);
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    vflux = v;   
    dynamic = v(end)*(1-v(end-2)) * v(end-3);
    fixed = v(end-1)*(v(end-2));
    vflux(end) = dynamic + fixed;
    v = vflux;
    fb = fbouaxialnd(u, q, w, v, x, t, mu, eta, uhat, n, tau, eta, mu, 0);
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
    vflux = v;   
    dynamic = v(end)*(1-v(end-2)) * v(end-3);
    fixed = v(end-1)*(v(end-2));
    vflux(end) = dynamic + fixed;
    v = vflux;
    f = eosnd(u, q, w, v, x, t, mu, eta);    
end

function f = sourcew(u, q, w, v, x, t, mu, eta)
    vflux = v;   
    dynamic = v(end)*(1-v(end-2)) * v(end-3);
    fixed = v(end-1)*(v(end-2));
    vflux(end) = dynamic + fixed;
    v = vflux;
    f = eosnd(u, q, w, v, x, t, mu, eta);        
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    vflux = v;   
    dynamic = v(end)*(1-v(end-2)) * v(end-3);
    fixed = v(end-1)*(v(end-2));
    vflux(end) = dynamic + fixed;
    v = vflux;
  fb = fbouhdgaxialnd(u, q, w, v, x, t, mu, eta, uhat, n, tau);
end

