function s = sourceaxial2d(u, q, w, v, x, t, mu, eta)
%SOURCEAXIAL2D Axisymmetric source term for 2D (z,r) compressible NS.
%   This routine adds the cylindrical geometric source terms for the
%   axisymmetric no-swirl equations written in the Exasim form
%
%       dU/dt + dFz/dz + dFr/dr = S .
%
%   It is designed to be consistent with fluxaxial2d.m and fluxcart2d.m.
%   If a Cartesian source function sourcecart2d.m exists (e.g. chemistry,
%   body-force terms, etc.), it is added automatically.
%
%   Coordinates: x(1)=z, x(2)=r.
%   Exasim convention: q = -grad(u).

ns = 5;
nch = ns + 3;

% Start with any Cartesian source already present (e.g. chemistry).
s = zeros(nch,1);
if exist('sourcend','file') == 2
    s = sourcend(u, q, w, v, x, t, mu, eta);
end

% Cylindrical radius.
r = x(2);
rinv = 1.0 / r;

% Add the scalar cylindrical-divergence contributions using the total
% radial flux returned by fluxaxial2d:
%
%   dU/dt + dFz/dz + (1/r)d(r Fr)/dr = ...
%       => dU/dt + dFz/dz + dFr/dr = ... - Fr/r
%
% This is correct for species, axial momentum, and energy. For the radial
% momentum equation there is an additional basis-vector term + (p-t_theta)/r,
% which is added below.
[f, p, ttt] = fluxaxial2d(u, q, w, v, x, t, mu, eta);
s = s - f(:,2) * rinv;
s(ns+2) = s(ns+2) + (p - ttt) * rinv;

end
