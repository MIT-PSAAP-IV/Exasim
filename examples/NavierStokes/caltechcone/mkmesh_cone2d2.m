function [mesh, X, Y] = mkmesh_cone2d2(porder, nx1, nx2, ny, dwall, theta_deg)

if nargin < 1, porder   = 2;      end
if nargin < 2, nx1      = 36;     end   % points in nose block
if nargin < 3, nx2      = 141;    end   % points in cone block
if nargin < 4, ny       = 101;    end   % wall-normal points
if nargin < 5, dwall    = .0002;   end   % near-wall spacing
if nargin < 6, theta_deg = 5;     end   % cone half-angle in degrees

theta = theta_deg*pi/180;         % convert to radians

% ---------------------------------------------------------
% Mesh stretching / geometric parameters
% ---------------------------------------------------------
s1 = 0.5;                         % streamwise stretching in nose block
s2 = 4;                           % streamwise stretching in cone block

rnose = 2.0e-3;                   % nose radius
Lcone = 0.99;                     % cone surface length after tangency point
R     = 0.04;                     % farfield distance in wall-normal direction
H     = 0.2;

Href = 1;
yref = [];%[0.4 0.15 0.06 0.015]*0.8;

% ---------------------------------------------------------
% Cone geometry
% Nose is a circular arc that ends tangent to the cone
% ---------------------------------------------------------
xt = rnose*(1 - sin(theta));      % tangency point x
yt = rnose*cos(theta);            % tangency point y

xend = xt + Lcone*cos(theta);     % cone end point on wall
yend = yt + Lcone*sin(theta);

Hfar = R*Href;
sEnd = Lcone;                     % tangent-coordinate length to outflow plane

% ---------------------------------------------------------
% Build reference mesh:
%   block 1: x in [0,1]   -> nose
%   block 2: x in [1,2]   -> straight cone
% ---------------------------------------------------------
x2 = loginc(linspace(1,2,nx2), s2);
[p2,t2,yv] = lesmesh2d_rect(Href, dwall, ny, x2, yref);

x1 = loginc(linspace(0,1,nx1), s1);
[p1,t1] = quadgrid(x1, yv);

[p,t] = connectmesh(p1, t1, p2', t2', 1e-10);

elemtype = 1;
nodetype = 0;

mesh = mkmesh(p, t, porder, {'true'}, elemtype, nodetype);

% =========================================================
% Map mesh vertices to physical space
% =========================================================
pnew = p;

ind1 = p(:,1) <= 1 + 1e-8;    % nose block
ind2 = p(:,1) >  1 - 1e-8;    % cone block
% =========================================================
% Map mesh vertices to physical space
% =========================================================
xi_v  = p(:,1);
eta_v = p(:,2);

pnew = zeros(size(p));

inose_v = (xi_v <= 1 + 1e-12);
icone_v = (xi_v >  1 + 1e-12);

% ---- nose block ----
phi = (pi/2 - theta) * xi_v(inose_v);
yn  = R * eta_v(inose_v);

pnew(inose_v,1) = rnose - (rnose + yn).*cos(phi);
pnew(inose_v,2) =         (rnose + yn).*sin(phi);

% ---- cone block ----
xi = xi_v(icone_v) - 1;  % [0, 1]
eta = eta_v(icone_v);    % [0, 1]

x1 = Lcone*xi;
y1 = (R + (H-R)*xi).*eta;

pnew(icone_v,1) = cos(theta).*x1 - sin(theta).*y1 + xt;
pnew(icone_v,2) = sin(theta).*x1 + cos(theta).*y1 + yt;

% =========================================================
% Map DG nodes to physical space
% IMPORTANT: use reference dgnodes, not overwritten dgx/dgy
% =========================================================
xi_dg  = mesh.dgnodes(:,1,:);
eta_dg = mesh.dgnodes(:,2,:);

dgx = zeros(size(xi_dg));
dgy = zeros(size(eta_dg));

inose_dg = (xi_dg <= 1 + 1e-12);
icone_dg = (xi_dg >  1 + 1e-12);

% ---- nose DG nodes ----
phi = (pi/2 - theta) * xi_dg(inose_dg);
yn  = R * eta_dg(inose_dg);

dgx(inose_dg) = rnose - (rnose + yn).*cos(phi);
dgy(inose_dg) =         (rnose + yn).*sin(phi);

% ---- cone DG nodes ----
xi = xi_dg(icone_dg) - 1;
eta = eta_dg(icone_dg);

x1 = Lcone*xi;
y1 = (R + (H-R)*xi).*eta;

dgx(icone_dg) = cos(theta).*x1 - sin(theta).*y1 + xt;
dgy(icone_dg)  = sin(theta).*x1 + cos(theta).*y1 + yt;

% =========================================================
% Build final mesh
% =========================================================
% =========================================================
% Build final mesh
% =========================================================
yshift = 1e-4;
pnew(:,2) = pnew(:,2) + yshift;
dgy = dgy + yshift;

mesh = mkmesh(pnew, t, porder, {'true'}, elemtype, nodetype);
mesh.dgnodes(:,1,:) = dgx;
mesh.dgnodes(:,2,:) = dgy;


% =========================================================
% Boundary definitions
%
% 1 = symmetry centerline
% 2 = inflow (outer circular arc)
% 3 = nose wall
% 4 = cone wall
% 5 = outflow
% 6 = freestream (outer straight boundary)
% =========================================================
tol = 1e-6;

xO = -Hfar;                    yO = yshift;
xL = 0;                        yL = yshift;
xC = xt;                       yC = yt + yshift;
xE = xend;                     yE = yend + yshift;
xA = xt - Hfar*sin(theta);     yA = yt + Hfar*cos(theta) + yshift;
xT = xE - H*Href*sin(theta);   yT = yE + H*Href*cos(theta);

rcap = @(p) sqrt((p(1,:)-rnose).^2 + (p(2,:)-yshift).^2);
scone = @(p)  (p(1,:)-xt)*cos(theta) + (p(2,:)-yC)*sin(theta);
ncone = @(p) -(p(1,:)-xt)*sin(theta) + (p(2,:)-yC)*cos(theta);
topline = @(p) (p(2,:)-yA)*(xT-xA) - (p(1,:)-xA)*(yT-yA);

mesh.boundaryexpr = {
  @(p) abs(p(2,:)-yO) < tol & (p(1,:) >= xO - tol) & (p(1,:) <= xL + tol), ...
  @(p) abs(rcap(p) - (rnose + Hfar)) < tol & (p(1,:) >= xO - tol) & (p(1,:) <= xA + tol), ...
  @(p) abs(rcap(p) - rnose) < tol & (p(1,:) >= xL - tol) & (p(1,:) <= xC + tol), ...
  @(p) abs(ncone(p)) < tol & (scone(p) >= -tol) & (scone(p) <= sEnd + tol), ...
  @(p) abs(scone(p) - sEnd) < tol & (ncone(p) >= -tol) & (ncone(p) <= H*Href + tol), ...
  @(p) abs(topline(p)) < 5e-6 & (p(1,:) >= min(xA,xT) - tol) & (p(1,:) <= max(xA,xT) + tol)
};

mesh.periodicboundary = [];
mesh.periodicexpr = {};

mesh.p = mesh.p';
mesh.t = mesh.t';
mesh.f = facenumbering(mesh.p, mesh.t, mesh.elemtype, ...
                       mesh.boundaryexpr, mesh.periodicexpr);

mesh.xpe   = mesh.plocal;
mesh.telem = mesh.tlocal;

X = mesh.dgnodes(:,1,:);
Y = mesh.dgnodes(:,2,:);

% =========================================================
% Plot
% =========================================================
figure(1); clf;
for ib = 1:6
    boundaryplot(mesh, ib); hold on;
end
plot(xt, yt, 'or', 'LineWidth', 2, 'MarkerSize', 8);
% plot(xend, yend, 'or', 'LineWidth', 2, 'MarkerSize', 8);
axis equal; axis tight;
% title(sprintf('Blunted cone mesh, \\theta = %.2f deg', theta_deg));

end
