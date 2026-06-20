function [mesh, X, Y] = mkmesh_cone2d(porder, nx1, nx2, ny, dwall, theta_deg)

if nargin < 1, porder   = 2;      end
if nargin < 2, nx1      = 27;     end   % points in nose block
if nargin < 3, nx2      = 261;    end   % points in cone block
if nargin < 4, ny       = 101;    end   % wall-normal points
if nargin < 5, dwall    = .0002;   end   % near-wall spacing
if nargin < 6, theta_deg = 5;     end   % cone half-angle in degrees

theta = theta_deg*pi/180;         % convert to radians

% ---------------------------------------------------------
% Mesh stretching / geometric parameters
% ---------------------------------------------------------
s1 = 0.5;                         % streamwise stretching in nose block
s2 = 3;                         % streamwise stretching in cone block

rnose = 2.0e-3;                   % nose radius
Lcone = 0.99;                     % axial cone length after tangency point
R     = 0.185;                     % farfield distance in wall-normal direction

Href = 1;
yref = []%[0.4 0.15 0.06 0.015]*0.8;

% ---------------------------------------------------------
% Cone geometry
% Nose is a circular arc that ends tangent to the cone
% ---------------------------------------------------------
xt = rnose*(1 - sin(theta));      % tangency point x
yt = rnose*cos(theta);            % tangency point y

xend = xt + Lcone;                % cone end point on wall
yend = yt + Lcone*tan(theta);

Hfar = R*Href;
sEnd = Lcone/cos(theta);          % tangent-coordinate length to outflow plane

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
xi = xi_v(icone_v) - 1;
yn = R * eta_v(icone_v);

xw = xt + Lcone*xi;
yw = yt + tan(theta)*(xw - xt);

pnew(icone_v,1) = xw - yn*sin(theta);
pnew(icone_v,2) = yw + yn*cos(theta);

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
yn = R * eta_dg(icone_dg);

xw = xt + Lcone*xi;
yw = yt + tan(theta)*(xw - xt);

dgx(icone_dg) = xw - yn*sin(theta);
dgy(icone_dg) = yw + yn*cos(theta);

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
% 1 = nose wall
% 2 = cone wall
% 3 = inflow (outer circular arc)
% 4 = freestream (outer straight boundary)
% 5 = outflow
% =========================================================
tolr = 1e-8;
toll = 1e-8;

tol = 1e-6;

xO = -Hfar;                    yO = yshift;
xL = 0;                        yL = yshift;
xC = xt;                       yC = yt + yshift;
xE = xt + Lcone;               yE = yt + Lcone*tan(theta) + yshift;
xA = xt - Hfar*sin(theta);     yA = yt + Hfar*cos(theta) + yshift;
xT = xE - Hfar*sin(theta);     yT = yE + Hfar*cos(theta);

rcap = @(p) sqrt((p(1,:)-rnose).^2 + (p(2,:)-yshift).^2);
mtop = (yT - yA)/(xT - xA);

mesh.boundaryexpr = {
  @(p) (p(2,:) < yO + tol) & (p(1,:) < xL + tol), ...
  @(p) (p(1,:) < xL + tol) & (rcap(p) > 2*rnose + tol), ...
  @(p) (p(1,:) > xL - tol) & (p(1,:) < xC + tol) & (rcap(p) < 2*rnose + tol), ...
  @(p) (p(2,:) < yE + tol) & (p(1,:) > xC - tol) & (p(1,:) < xE + tol), ...
  @(p) abs((p(1,:)-xt)*cos(theta) + (p(2,:)-(yt+yshift))*sin(theta) - sEnd) < tol, ...
  @(p) abs((p(2,:)-yA) - mtop*(p(1,:)-xA)) < 5e-6
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
% for ib = 1:length(mesh.boundaryexpr)
%     boundaryplot(mesh, ib); hold on;
% end
% plot(xt, yt, 'or', 'LineWidth', 2, 'MarkerSize', 8);
% plot(xend, yend, 'or', 'LineWidth', 2, 'MarkerSize', 8);
% axis equal; axis tight;
% title(sprintf('Blunted cone mesh, \\theta = %.2f deg', theta_deg));

end