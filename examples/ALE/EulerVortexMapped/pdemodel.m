function pde = pdemodel
pde.mass = @mass;
pde.flux = @flux;
pde.source = @source;
pde.fbou = @fbou;
pde.fbouhdg = @fbouhdg;
pde.ubou = @ubou;
pde.initu = @initu;
pde.visscalars = @visscalars;
pde.visvectors = @visvectors;
pde.qoivolume = @qoivolume;
end

% -------------------------------------------------------------------------
% ALE mapping: paper sinusoidal deformation (20x15 domain)
%   x1 = X1 + 2.0*sin(pi*X1/10)*sin(pi*X2/7.5)*sin(2*pi*t/t0)
%   x2 = X2 + 1.5*sin(pi*X1/10)*sin(pi*X2/7.5)*sin(4*pi*t/t0)
%   t0 = sqrt(10^2 + 5^2) = sqrt(125)
%
% Identity at t=0 and t=t0; vanishes on all domain boundaries.
% -------------------------------------------------------------------------

function xp = ale_mapping(x, t, mu)
    t0 = sqrt(sym(125));
    pi_ = sym(pi);
    X1 = x(1); X2 = x(2);

    sX = sin(pi_*X1/sym(10)) * sin(pi_*X2/sym(7.5));

    xp = [X1 + sym(2.0)*sX*sin(sym(2)*pi_*t/t0);
          X2 + sym(1.5)*sX*sin(sym(4)*pi_*t/t0)];
end

function G = deformation_gradient(x, t, mu)
    t0 = sqrt(sym(125));
    pi_ = sym(pi);
    X1 = x(1); X2 = x(2);

    cosX1 = cos(pi_*X1/sym(10));
    sinX1 = sin(pi_*X1/sym(10));
    cosX2 = cos(pi_*X2/sym(7.5));
    sinX2 = sin(pi_*X2/sym(7.5));

    s2 = sin(sym(2)*pi_*t/t0);
    s4 = sin(sym(4)*pi_*t/t0);

    G11 = sym(1) + (pi_/sym(5))*cosX1*sinX2*s2;
    G12 = (sym(2)*pi_/sym(7.5))*sinX1*cosX2*s2;
    G21 = (sym(1.5)*pi_/sym(10))*cosX1*sinX2*s4;
    G22 = sym(1) + (pi_/sym(5))*sinX1*cosX2*s4;

    G = [G11 G12; G21 G22];
end

function Jdet = mapping_jacobian(x, t, mu)
    G = deformation_gradient(x, t, mu);
    Jdet = G(1,1)*G(2,2) - G(1,2)*G(2,1);
end

function vm = mesh_velocity(x, t, mu)
    t0 = sqrt(sym(125));
    pi_ = sym(pi);
    X1 = x(1); X2 = x(2);

    sX = sin(pi_*X1/sym(10)) * sin(pi_*X2/sym(7.5));

    c2 = cos(sym(2)*pi_*t/t0);
    c4 = cos(sym(4)*pi_*t/t0);

    vm = [sym(4)*pi_/t0 * sX * c2;
          sym(6)*pi_/t0 * sX * c4];
end

function dJ = dJ_dt(x, t, mu)
    % Time derivative of det(G) — needed for GCL source term
    t0 = sqrt(sym(125));
    pi_ = sym(pi);
    X1 = x(1); X2 = x(2);

    cosX1 = cos(pi_*X1/sym(10));
    sinX1 = sin(pi_*X1/sym(10));
    cosX2 = cos(pi_*X2/sym(7.5));
    sinX2 = sin(pi_*X2/sym(7.5));

    s2 = sin(sym(2)*pi_*t/t0);
    c2 = cos(sym(2)*pi_*t/t0);
    s4 = sin(sym(4)*pi_*t/t0);
    c4 = cos(sym(4)*pi_*t/t0);

    % G and its time derivative
    G11 = sym(1) + (pi_/sym(5))*cosX1*sinX2*s2;
    G12 = (sym(2)*pi_/sym(7.5))*sinX1*cosX2*s2;
    G21 = (sym(1.5)*pi_/sym(10))*cosX1*sinX2*s4;
    G22 = sym(1) + (pi_/sym(5))*sinX1*cosX2*s4;

    w2 = sym(2)*pi_/t0;
    w4 = sym(4)*pi_/t0;

    dG11 = (pi_/sym(5))*cosX1*sinX2 * w2*c2;
    dG12 = (sym(2)*pi_/sym(7.5))*sinX1*cosX2 * w2*c2;
    dG21 = (sym(1.5)*pi_/sym(10))*cosX1*sinX2 * w4*c4;
    dG22 = (pi_/sym(5))*sinX1*cosX2 * w4*c4;

    % dJ/dt = dG11/dt * G22 + G11 * dG22/dt - dG12/dt * G21 - G12 * dG21/dt
    dJ = dG11*G22 + G11*dG22 - dG12*G21 - G12*dG21;
end

% -------------------------------------------------------------------------
% PDE functions for the ALE Euler equations
%   d(J*u)/dt + div_X(F_ref) = 0
%   F_ref = (F_phys - u * v_mesh^T) * cof(G)
%   cof(G) = J * G^{-T} = [G22, -G21; -G12, G11]
% -------------------------------------------------------------------------

function m = mass(u, q, w, v, x, t, mu, eta)
    Jdet = mapping_jacobian(x, t, mu);
    m = sym([Jdet; Jdet; Jdet; Jdet]);
end

function f = flux(u, q, w, v, x, t, mu, eta)
    gam = mu(1);
    gam1 = gam - 1.0;
    r = u(1);
    ru = u(2);
    rv = u(3);
    rE = u(4);
    r1 = 1/r;
    uv = ru*r1;
    vv = rv*r1;
    ke = 0.5*(uv*uv+vv*vv);
    p = gam1*(rE - r*ke);
    h = (rE + p)*r1;

    % physical Euler flux
    f_phys = [ru, ru*uv+p, rv*uv, ru*h, rv, ru*vv, rv*vv+p, rv*h];
    f_phys = reshape(f_phys, [4 2]);

    % ALE correction with Piola transform
    G = deformation_gradient(x, t, mu);
    G11 = G(1,1); G12 = G(1,2); G21 = G(2,1); G22 = G(2,2);
    cof = [G22, -G21; -G12, G11];
    vm = mesh_velocity(x, t, mu);
    f_ref = (f_phys - u * vm.') * cof;

    f = f_ref;
end

function s = source(u, q, w, v, x, t, mu, eta)
    % GCL source: ∂(J·u)/∂t = J·∂u/∂t + u·∂J/∂t
    % Exasim solves J·∂u/∂t + div(F_ref) = source, so we need
    % source = -u·∂J/∂t to recover the conservative form
    dJ = dJ_dt(x, t, mu);
    s = -u * dJ;
end

function fb = fbou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    f = flux(u, q, w, v, x, t, mu, eta);
    fb = f(:,1)*n(1) + f(:,2)*n(2) + tau*(u-uhat);
end

function ub = ubou(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    ub = sym([0.0; 0.0; 0.0; 0.0]);
end

function fb = fbouhdg(u, q, w, v, x, t, mu, eta, uhat, n, tau)
    fb0 = tau*(sym(0.0)-uhat(1));
    fb = [fb0; fb0; fb0; fb0];
end

% -------------------------------------------------------------------------
% Initial and exact solution: isentropic Euler vortex
% Evaluated at physical coordinates x_phys = ale_mapping(x, t, mu)
% Parameters: mu = [gam, Minf, eps, rc, x0, y0]
% -------------------------------------------------------------------------

function u0 = initu(x, mu, eta)
    u0 = vortex(x, sym(0), mu);
end

function ue = vortex(x, t, mu)
    % x is reference coordinate X; map to physical coordinate
    xp = ale_mapping(x, t, mu);

    gam = mu(1);
    gam1 = gam - 1.0;
    Minf = mu(2);
    eps = mu(3);
    rc = mu(4);
    x0 = mu(5);
    y0 = mu(6);

    costh = 2/sqrt(sym(5));
    sinth = 1/sqrt(sym(5));
    ubar = Minf*costh;
    vbar = Minf*sinth;

    xcent = x0 + ubar*t;
    ycent = y0 + vbar*t;

    dx = xp(1) - xcent;
    dy = xp(2) - ycent;

    f = 1.0 - (dx*dx + dy*dy)/(rc*rc);
    coeff = gam1*eps*eps/(8*sym(pi)*sym(pi));
    T = 1.0 - coeff*exp(f);
    rho = T^(1/gam1);
    u = ubar - eps*dy/(2*sym(pi)*rc)*exp(f/2);
    v = vbar + eps*dx/(2*sym(pi)*rc)*exp(f/2);
    p = T^(gam/gam1)/gam;
    rhoE = p/gam1 + 0.5*rho*(u*u + v*v);

    ue = [rho; rho*u; rho*v; rhoE];
end

% -------------------------------------------------------------------------
% QoI: L2 error norm (integrated in reference domain)
% -------------------------------------------------------------------------

function q = qoivolume(u, q, w, v, x, t, mu, eta)
    Jdet = mapping_jacobian(x, t, mu);
    ue = vortex(x, t, mu);
    err = u - ue;
    q = Jdet * (err(1)^2 + err(2)^2 + err(3)^2 + err(4)^2);
end

% -------------------------------------------------------------------------
% Visualization: density, pressure, density error (scalars)
%                mesh displacement vector (for Paraview Warp By Vector)
% -------------------------------------------------------------------------

function s = visscalars(u, q, w, v, x, t, mu, eta)
    gam = mu(1);
    gam1 = gam - 1.0;
    r = u(1);
    ru = u(2);
    rv = u(3);
    rE = u(4);
    r1 = 1/r;
    uv = ru*r1;
    vv = rv*r1;
    p = gam1*(rE - r*0.5*(uv*uv+vv*vv));

    ue = vortex(x, t, mu);
    err = r - ue(1);

    s = [r; p; err];
end

function s = visvectors(u, q, w, v, x, t, mu, eta)
    % mesh displacement: x_phys - X  (for Paraview Warp By Vector)
    xp = ale_mapping(x, t, mu);
    dx = xp(1) - x(1);
    dy = xp(2) - x(2);
    s = [dx; dy];
end
