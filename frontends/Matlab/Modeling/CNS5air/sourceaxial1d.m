function s = sourceaxial1d(u, q, w, v, x, t, mu, eta)
%SOURCEAXIAL1D Source term for the 1D axial reduction in cylindrical coordinates.
%   This routine is consistent with fluxaxial1d.m, which is the strict 1D
%   reduction of fluxaxial2d.m obtained by assuming
%
%       u_r = 0,   d/dr(.) = 0 .
%
%   Under this reduction, the cylindrical geometric source terms vanish:
%
%     - the generic cylindrical-divergence term  -F_r/r  is zero because the
%       retained radial flux F_r is zero in the 1D model;
%     - there is no radial-momentum equation, so there is no extra
%       basis-vector correction +(p - tau_{theta theta})/r.
%
%   Therefore sourceaxial1d only returns any pre-existing non-geometric
%   source terms (e.g. chemistry, body forces) supplied by the base model.
%
%   Coordinate: x(1)=z.
%   State: [rho_1,...,rho_5, rho*u_z, rhoE].
%   Exasim convention: q = -grad(u).

ns = 5;
nch = ns + 2;

% Start with any pre-existing base source (chemistry, body forces, ...).
s = zeros(nch,1);
if exist('sourcend','file') == 2
    s = sourcend(u, q, w, v, x, t, mu, eta);
end

end
