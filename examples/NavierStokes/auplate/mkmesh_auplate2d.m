function [mesh, X, Y, geom] = mkmesh_auplate2d(porder)

% =========================================================
% Minimal new-Exasim mesh for rounded-nose flat plate
%
% Geometry:
%   Rounded circular nose, radius rnose
%   Completely flat horizontal plate to xEnd
%
% Boundary numbering:
%   1 = symmetry centerline upstream of nose
%   2 = inflow / outer nose arc
%   3 = rounded nose wall
%   4 = flat plate wall
%   5 = outflow
%   6 = freestream straight-line top
% =========================================================

if nargin < 1
    porder = 2;
end

% ---------------------------------------------------------
% Mesh resolution / stretching
% ---------------------------------------------------------
nxNose = 15;
nxBody = 121;
ny     = 61;
dwall = .0003;

sNose = 0.5;
sBody = 6;

elemtype = 1;
nodetype = 0;

% ---------------------------------------------------------
% Geometry constants
% ---------------------------------------------------------
rnose = 8.0e-5;    % 0.8 mm nose radius
xEnd  = 0.600;     % 600 mm total length from leading tip

% For the upper-half rounded flat plate:
% nose center is at (rnose, yshift)
% flat wall begins at x = rnose, y = rnose + yshift
xFlat0 = rnose;
yFlat  = rnose;

% Straight-line top boundary controls
Hnose = 0.04;
Hend  = 0.255;

Href   = 1.0;
yshift = 1e-4;
yref   = [];

plotMesh = true;

% ---------------------------------------------------------
% Derived geometry
% ---------------------------------------------------------
yFlatS = yFlat + yshift;

Lbody = xEnd - xFlat0;

% Top boundary starts above nose/body tangency
xA = xFlat0;
yA = yFlat + Hnose + yshift;

% Top boundary ends above outflow
xT = xEnd;
yT = yFlat + Hend + yshift;

% ---------------------------------------------------------
% Build reference mesh:
%   xi in [0,1] -> circular nose
%   xi in [1,2] -> flat body
% ---------------------------------------------------------
[~,~,yv] = lesmesh2d_rect(Href, dwall, ny, [0 1], yref);

xN =       loginc(linspace(0,1,nxNose), sNose);
xB = 1.0 + loginc(linspace(0,1,nxBody), sBody);

[pN, tN] = quadgrid(xN, yv);
[pB, tB] = quadgrid(xB, yv);

[p, t] = connectmesh(pN, tN, pB, tB, 1e-10);

% Temporary reference mesh for DG nodes
mesh0 = mkmesh(p, t, porder, {'true'}, elemtype, nodetype);

% ---------------------------------------------------------
% Map vertices
% ---------------------------------------------------------
pnew = zeros(size(p));
[pnew(:,1), pnew(:,2)] = map_to_physical(p(:,1), p(:,2));

% ---------------------------------------------------------
% Map DG nodes from reference coordinates
% ---------------------------------------------------------
xi_dg  = mesh0.dgnodes(:,1,:);
eta_dg = mesh0.dgnodes(:,2,:);

[dgx, dgy] = map_to_physical(xi_dg, eta_dg);

% ---------------------------------------------------------
% Build final physical mesh
% ---------------------------------------------------------
mesh = mkmesh(pnew, t, porder, {'true'}, elemtype, nodetype);

mesh.dgnodes(:,1,:) = dgx;
mesh.dgnodes(:,2,:) = dgy;

% =========================================================
% Boundary definitions
% =========================================================
tol = 1.0e-6;

% Radius from circular nose center
rcap = @(p) sqrt((p(1,:) - rnose).^2 + ...
                 (p(2,:) - yshift).^2);

xO = -Hnose;
xL = 0.0;

% Straight top line from A to T
topline = @(p) (p(2,:) - yA).*(xT - xA) - ...
               (p(1,:) - xA).*(yT - yA);

tauTop = @(p) ((p(1,:) - xA).*(xT - xA) + ...
               (p(2,:) - yA).*(yT - yA)) ./ ...
              ((xT - xA)^2 + (yT - yA)^2);

% Outflow line from wall end to top point
outflowline = @(p) (p(2,:) - yFlatS).*(xT - xEnd) - ...
                   (p(1,:) - xEnd  ).*(yT - yFlatS);

tauOut = @(p) ((p(1,:) - xEnd  ).*(xT - xEnd) + ...
               (p(2,:) - yFlatS).*(yT - yFlatS)) ./ ...
              ((xT - xEnd)^2 + (yT - yFlatS)^2);

flatWall = @(p) abs(p(2,:) - yFlatS) < tol & ...
                p(1,:) >= xFlat0 - tol & ...
                p(1,:) <= xEnd   + tol;

mesh.boundaryexpr = {
    @(p) abs(p(2,:) - yshift) < tol & ...
         p(1,:) >= xO - tol & p(1,:) <= xL + tol, ...

    @(p) abs(rcap(p) - (rnose + Hnose)) < 5*tol & ...
         p(1,:) >= xO - tol & p(1,:) <= xA + tol & ...
         p(2,:) >= yshift - tol & p(2,:) <= yA + tol, ...

    @(p) abs(rcap(p) - rnose) < tol & ...
         p(1,:) >= -tol & p(1,:) <= xFlat0 + tol & ...
         p(2,:) >= yshift - tol & p(2,:) <= yFlatS + tol, ...

    @(p) flatWall(p), ...

    @(p) abs(outflowline(p)) < 5*tol & ...
         tauOut(p) >= -tol & tauOut(p) <= 1 + tol, ...

    @(p) abs(topline(p)) < 5*tol & ...
         tauTop(p) >= -tol & tauTop(p) <= 1 + tol
};

mesh.periodicboundary = [];
mesh.periodicexpr = {};

% ---------------------------------------------------------
% New Exasim mesh storage format
% ---------------------------------------------------------
mesh.p = mesh.p';
mesh.t = mesh.t';

mesh.f = facenumbering(mesh.p, mesh.t, mesh.elemtype, ...
                       mesh.boundaryexpr, mesh.periodicexpr);

mesh.xpe   = mesh.plocal;
mesh.telem = mesh.tlocal;

X = mesh.dgnodes(:,1,:);
Y = mesh.dgnodes(:,2,:);

% ---------------------------------------------------------
% Geometry output
% ---------------------------------------------------------
geom.rnose = rnose;

geom.xTip = 0.0;
geom.yTip = 0.0;

geom.xTangency = xFlat0;
geom.yTangency = yFlat;

geom.xFlatStart = xFlat0;
geom.yFlat      = yFlat;

geom.xEnd = xEnd;
geom.yEnd = yFlat;
geom.DEnd = 2*yFlat;

geom.Lnose = 0.5*pi*rnose;
geom.Lbody = Lbody;
geom.LwallIncludingNose = geom.Lnose + geom.Lbody;

geom.Hnose = Hnose;
geom.Hend  = Hend;

geom.xTopStart = xA;
geom.yTopStart = yA;
geom.xTopEnd   = xT;
geom.yTopEnd   = yT;

geom.yshift = yshift;

% ---------------------------------------------------------
% Plot
% ---------------------------------------------------------
if plotMesh
    figure(1); clf;

    for ib = 1:6
        boundaryplot(mesh, ib); hold on;
    end

    phi = linspace(0, pi/2, 200);
    xwallN = rnose - rnose*cos(phi);
    ywallN = rnose*sin(phi) + yshift;

    plot([xwallN xEnd], ...
         [ywallN yFlatS], ...
         '-r', 'LineWidth', 2);

    plot([0 xFlat0 xEnd], ...
         [yshift yFlatS yFlatS], ...
         'or', 'LineWidth', 2, 'MarkerSize', 7);

    axis equal;
    axis tight;
    title(sprintf('Rounded-nose flat plate mesh, R_n = %.3f mm, L = %.1f mm', ...
          rnose*1e3, xEnd*1e3));
    drawnow;
end

% =========================================================
% Mapping from reference coordinates to physical coordinates
% =========================================================
    function [xphys, yphys] = map_to_physical(xi, eta)

        xphys = zeros(size(xi));
        yphys = zeros(size(xi));

        inose = xi <= 1.0 + 1e-12;
        ibody = xi >  1.0 + 1e-12;

        % -------------------------------------------------
        % Nose cap: concentric circular quarter-cap
        %
        % eta = 0 -> rounded nose wall, radius rnose
        % eta = 1 -> outer nose boundary, radius rnose + Hnose
        %
        % phi = 0    -> leading tip / symmetry line
        % phi = pi/2 -> tangency with flat plate
        % -------------------------------------------------
        phi = (pi/2).*xi(inose);
        h   = Hnose.*eta(inose);

        xphys(inose) = rnose - (rnose + h).*cos(phi);
        yphys(inose) =          (rnose + h).*sin(phi) + yshift;

        % -------------------------------------------------
        % Flat body block
        %
        % eta = 0 -> flat wall
        % eta = 1 -> straight top line
        % -------------------------------------------------
        u = xi(ibody) - 1.0;

        xw = xFlat0 + Lbody.*u;
        yw = yFlatS + 0.*u;

        lam = (xw - xFlat0)./Lbody;

        xtop = xA + lam.*(xT - xA);
        ytop = yA + lam.*(yT - yA);

        xphys(ibody) = (1 - eta(ibody)).*xw + eta(ibody).*xtop;
        yphys(ibody) = (1 - eta(ibody)).*yw + eta(ibody).*ytop;
    end

end