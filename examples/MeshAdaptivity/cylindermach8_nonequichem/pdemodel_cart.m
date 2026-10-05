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
    pde.surfacequantities = @surfacequantities;
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
    sensor = limiting(compression*tanh(mu(end-2)*v(1)), ...
                      0, mu(end-3), 1.0e3, 0);
    [~, Mw, ~] = thermodynamicsModels();
    pressureNondimensional = pressure(mu(4)*w(1), mu(1)*u(1:5), Mw)/mu(3);
    f = [sensor; pressureNondimensional];
end

function s = visscalars(u, q, w, v, x, t, mu, eta) %#ok<INUSD>
    [~, Mw, ~] = thermodynamicsModels();
    s = pressure(mu(4)*w(1), mu(1)*u(1:5), Mw)/mu(3);
end

function s = surfacequantities(u, q, w, v, x, t, mu, eta, uhat, n, tau) %#ok<INUSD>
% Pointwise catalytic-wall outputs evaluated by the backend on pde.ibs.
%
% s(1) = Cp = (p - p_inf)/(0.5*rho_inf*|u_inf|^2).
% s(2) = Cf = t dot (tau_v n + HDG momentum penalty) /
%        (0.5*rho_inf*|u_inf|^2), with t=[-n_y,n_x].
% s(3) = Cq = (kappa grad(T) dot n + HDG energy penalty) /
%        (rho_inf*|u_inf|^3).  Positive Cq follows the saved outward
%        boundary normal and is consistent with the sign convention used
%        by the compressible Navier--Stokes flux in fluxcart2d.

ns = 5;
ndim = numel(x);
nch = ns + ndim + 1;
[~, Mw, ~] = thermodynamicsModels();

rhoInf = sum(eta(1:ns));
uInf = eta(ns+1)/rhoInf;
vInf = eta(ns+2)/rhoInf;
speedInf2 = uInf*uInf + vInf*vInf;
qdyn = 0.5*rhoInf*speedInf2;
qheatref = rhoInf*speedInf2*sqrt(speedInf2);
pInf = pressure(mu(4), mu(1)*eta(1:ns), Mw)/mu(3);

rho = sum(uhat(1:ns));
uv = uhat(ns+1)/rho;
vv = uhat(ns+2)/rho;
rhoE = uhat(ns+3);
rho_i_dim = mu(1)*uhat(1:ns);

T_dim = mu(4)*w(1);
p = pressure(T_dim, mu(1)*uhat(1:ns), Mw)/mu(3);

drho_dx_i = -q(1:ns);
drhou_dx = -q(ns+1);
drhov_dx = -q(ns+2);
drhoE_dx = -q(ns+3);
drho_dy_i = -q((nch+1):(nch+ns));
drhou_dy = -q(nch+ns+1);
drhov_dy = -q(nch+ns+2);
drhoE_dy = -q(nch+ns+3);
drho_dx = sum(drho_dx_i);
drho_dy = sum(drho_dy_i);

drhoe_dx = drhoE_dx - (uv*drhou_dx + vv*drhov_dx) ...
    + 0.5*(uv*uv + vv*vv)*drho_dx;
drhoe_dy = drhoE_dy - (uv*drhou_dy + vv*drhov_dy) ...
    + 0.5*(uv*uv + vv*vv)*drho_dy;

du_dx = (drhou_dx - uv*drho_dx)/rho;
du_dy = (drhou_dy - uv*drho_dy)/rho;
dv_dx = (drhov_dx - vv*drho_dx)/rho;
dv_dy = (drhov_dy - vv*drho_dy)/rho;

[dT_drho_i_dim, dT_drhoe_dim, ~, ~, mu_d_dim, kappa_dim] = transportcoefficients(T_dim, rho_i_dim);
dT_drho_i = dT_drho_i_dim * mu(1) / mu(4);
dT_drhoe = dT_drhoe_dim * mu(3) / mu(4);
dT_dx = sum(dT_drho_i .* drho_dx_i) + dT_drhoe * drhoe_dx;
dT_dy = sum(dT_drho_i .* drho_dy_i) + dT_drhoe * drhoe_dy;

mu_d = mu_d_dim / mu(5);
kappa = kappa_dim / mu(6);
Re = mu(11);
Pr = mu(10);
Ec = mu(9);

txx = mu_d * (2.0/3.0) * (2.0*du_dx - dv_dy) / Re;
txy = mu_d * (du_dy + dv_dx) / Re;
tyy = mu_d * (2.0/3.0) * (2.0*dv_dy - du_dx) / Re;

nx = n(1);
ny = n(2);
tx = -ny;
ty = nx;
tractionx = txx*nx + txy*ny + tau_entry(tau,ns+1)*(u(ns+1)-uhat(ns+1));
tractiony = txy*nx + tyy*ny + tau_entry(tau,ns+2)*(u(ns+2)-uhat(ns+2));
tauTangential = tx*tractionx + ty*tractiony;

conductiveWallFlux = kappa*(dT_dx*nx + dT_dy*ny)/(Re*Pr*Ec) ...
    + tau_entry(tau,nch)*(u(nch)-uhat(nch));

Cp = (p - pInf)/qdyn;
Cf = tauTangential/qdyn;
Cq = conductiveWallFlux/qheatref;
s = [Cp; Cf; Cq];
end

function va = artificialviscosity(v, mu)
    av = (mu(end-1) + mu(end)*v(2))*tanh(mu(end-2)*v(1));
    va = [av; av];
end

function tauc = tau_entry(tau, idx)
if numel(tau) == 1
    tauc = tau;
else
    tauc = tau(idx);
end
end
