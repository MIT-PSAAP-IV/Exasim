function s = sourceaxial1d_quasi1d(u, q, w, v, x, t, mu, eta)
%SOURCEAXIAL1D_QUASI1D Quasi-1D nozzle-area source for five-species air.
%   Coordinate: x(1) = z.
%   State stored in u:
%
%       u = [rho_1*A, ..., rho_5*A, rho*u_z*A, rhoE*A]^T.
%
%   This routine is consistent with fluxaxial1d_quasi1d.m and supplies the
%   quasi-1D nozzle source terms in conservative area-weighted form:
%
%       d(AU)/dt + d(AF)/dz = A*S_cart + S_nozzle,
%
%   with
%
%       S_nozzle = [0, ..., 0, p*dA/dz, 0]^T.
%
%   Here p is the nondimensional pressure corresponding to the physical
%   state U = u / A.
%
%   The helper function
%
%       [A, Aderiv] = nozzlearea(z)
%
%   is assumed available. If your function is named differently, change the
%   single line below accordingly.

ns = 5;
nch = ns + 2;

z = x(1);
[A, Aderiv] = nozzlearea(z);
Ainv = 1.0 / A;

% Start with any base Cartesian source (e.g. chemistry) and convert it to
% area-weighted form.
s = zeros(nch,1);
if exist('sourcecart1d','file') == 2
    % sourcecart1d expects physical conservative variables, not area-weighted.
    uphys = u * Ainv;
    qphys = zeros(size(q));
    % Recover physical z-derivatives if a base source uses q.
    for i = 1:nch
        qphys(i) = -( (-q(i)) - Aderiv * uphys(i) ) * Ainv;
    end
    s = A * sourcecart1d(uphys, qphys, w, v, x, t, mu, eta);
elseif exist('sourcend','file') == 2
    uphys = u * Ainv;
    qphys = zeros(size(q));
    for i = 1:nch
        qphys(i) = -( (-q(i)) - Aderiv * uphys(i) ) * Ainv;
    end
    s = A * sourcend(uphys, qphys, w, v, x, t, mu, eta);
end

% Pressure from physical state U = u / A.
[~, Mw, ~] = thermodynamicsModels();
rho_i = u(1:ns) * Ainv;
rho_i_dim = rho_i * mu(1);
T_dim = w(1) * mu(4);
p_dim = pressure(T_dim, rho_i_dim, Mw);
p = p_dim / mu(3);

% Quasi-1D nozzle source: only axial momentum receives p*A_z.
s(ns+1) = s(ns+1) + p * Aderiv;

end
