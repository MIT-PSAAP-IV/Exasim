function [mesh, X, Y, geom] = mkmesh_auplate2d4UR(porder)

% =========================================================
% Minimal new-Exasim mesh for rounded-nose flat plate
% Includes yref transition for targeted refinement
%
% *** COARSENED VERSION ***
% This mesh is a 4x-per-direction coarsening of the original
% mkmesh_auplate2d4dns mesh (nxNose=35, nxBody=221, ny=105,
% dwall=0.00015). Each quad element here, when uniformly
% subdivided into 4x4 = 16 smaller elements, reproduces a mesh
% very close to the original:
%   ny:      lesmesh2d_rect treats ny as an ELEMENT count directly
%             (geometric BL stretching, h_i = dwall*rat^i, solved so
%             the sum spans dlay). Original ny=105 elements / 4 =
%             26.25 -> rounded to 26 -> refines back to 104 (orig.
%             was 105; off by only 1 element). Verified numerically:
%             solving for rat at (dwall=0.0006, ny=26) gives
%             rat_coarse ~ 1.2639, vs rat_orig^4 ~ 1.2553 for the
%             original (dwall=0.00015, ny=105) -> after a uniform
%             4-way split the y-position profile matches the
%             original to within ~0.7% of the BL thickness.
%   nxBody:  220 elements / 4 = 55  -> refines back to 220 (exact)
%             [nxBody/nxNose are linspace POINT counts, elements = n-1]
%   nxNose:   34 elements / 4 = 8.5 -> rounded up to 9 -> refines
%             to 36 (orig. was 34; off by 2 points on the nose arc,
%             not exactly divisible by 4)
%   dwall:   scaled by 4x (0.00015 -> 0.0006) so that after the
%             y-direction is quartered, the near-wall spacing
%             returns to the original 0.00015. Note the refined
%             mesh grows in "steps of 4" (4 equal sub-cells, then
%             a jump) rather than smoothly every cell, since the
%             16-way split is a uniform subdivision, not a re-solve
%             of the geometric series.
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
nxNose = 10;    % coarse: 9 elements (orig 35 pts / 34 elem, not exactly /4 -> rounded up)
nxBody = 56;    % coarse: 55 elements (orig 221 pts / 220 elem, exact /4)
ny     = 26;    % coarse: 26 ELEMENTS directly (lesmesh2d_rect's ny is an element
                % count, not a point count); orig ny=105 elements / 4 = 26.25 ->
                % rounded to 26 -> refines back to 104 (orig 105, off by 1)
dwall  = .0006; % coarse: 4x original 0.00015, so refined near-wall spacing matches original

sNose = 0.5;
sBody = 2.75;   % Original continuous left-to-right bunching

elemtype = 1;
nodetype = 0;

% ---------------------------------------------------------
% Geometry constants & Cutoff
% ---------------------------------------------------------
rnose   = 8.0e-5;
xEnd    = 0.5;
xCutoff = 0.280;     % Location where yref switches

xFlat0 = rnose;
yFlat  = rnose;

Hnose = 0.04;
Hend  = 0.135;

Href   = 1.0;
yshift = 1e-4;

% Differentiated y-refinements
yref1 = [0.4 0.15 0.06 .035 0.01]*0.8;          % Laminar (4 values)
yref2 = [0.4 0.15 0.06 .035 0.01]*0.8;    % Turbulent (5 values)

plotMesh = true;

% ---------------------------------------------------------
% Derived geometry
% ---------------------------------------------------------
yFlatS = yFlat + yshift;
Lbody  = xEnd - xFlat0;

xA = xFlat0;
yA = yFlat + Hnose + yshift;
xT = xEnd;
yT = yFlat + Hend + yshift;

% ---------------------------------------------------------
% Build reference meshes
% ---------------------------------------------------------
% Generate the FULL original horizontal distribution
xB_full = 1.0 + loginc(linspace(0,1,nxBody), sBody);

% Map the reference distribution to physical space to find the cutoff index
x_phys_B = xFlat0 + Lbody .* (xB_full - 1.0);

% Find transition index
split_idx = find(x_phys_B >= xCutoff, 1);

% Slice the distribution
xB1 = xB_full(1:split_idx);
xB2 = xB_full(split_idx:end);

% ---------------------------------------------------------
% Extra clustering approaching transition from LEFT
% ---------------------------------------------------------
sTrans = 0.2;

q1 = (xB1 - xB1(1)) / (xB1(end) - xB1(1));

% Mirror loginc so spacing decreases toward xCutoff
q1 = 1 - loginc(1 - q1, sTrans);

xB1 = xB1(1) + (xB1(end) - xB1(1))*q1;

% Apply the respective yrefs to each slice
[pB1, tB1, yv] = lesmesh2d_rect(Href, dwall, ny, xB1, yref1);
[pB2, tB2, ~]  = lesmesh2d_rect(Href, dwall, ny, xB2, yref2);

% Nose block
xN = loginc(linspace(0,1,nxNose), sNose);
[pN, tN] = quadgrid(xN, yv);

% Connect all blocks sequentially
[p12, t12] = connectmesh(pN, tN, pB1', tB1', 1e-10);
[p, t]     = connectmesh(p12, t12, pB2', tB2', 1e-10);

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
rcap = @(p) sqrt((p(1,:) - rnose).^2 + (p(2,:) - yshift).^2);

xO = -Hnose;
xL = 0.0;

topline = @(p) (p(2,:) - yA).*(xT - xA) - (p(1,:) - xA).*(yT - yA);
tauTop  = @(p) ((p(1,:) - xA).*(xT - xA) + (p(2,:) - yA).*(yT - yA)) ./ ((xT - xA)^2 + (yT - yA)^2);

outflowline = @(p) (p(2,:) - yFlatS).*(xT - xEnd) - (p(1,:) - xEnd).*(yT - yFlatS);
tauOut      = @(p) ((p(1,:) - xEnd).*(xT - xEnd) + (p(2,:) - yFlatS).*(yT - yFlatS)) ./ ((xT - xEnd)^2 + (yT - yFlatS)^2);

flatWall = @(p) abs(p(2,:) - yFlatS) < tol & p(1,:) >= xFlat0 - tol & p(1,:) <= xEnd + tol;

mesh.boundaryexpr = {
    @(p) abs(p(2,:) - yshift) < tol & p(1,:) >= xO - tol & p(1,:) <= xL + tol, ...
    @(p) abs(rcap(p) - (rnose + Hnose)) < 5*tol & p(1,:) >= xO - tol & p(1,:) <= xA + tol & p(2,:) >= yshift - tol & p(2,:) <= yA + tol, ...
    @(p) abs(rcap(p) - rnose) < tol & p(1,:) >= -tol & p(1,:) <= xFlat0 + tol & p(2,:) >= yshift - tol & p(2,:) <= yFlatS + tol, ...
    @(p) flatWall(p), ...
    @(p) abs(outflowline(p)) < 5*tol & tauOut(p) >= -tol & tauOut(p) <= 1 + tol, ...
    @(p) abs(topline(p)) < 5*tol & tauTop(p) >= -tol & tauTop(p) <= 1 + tol
};

mesh.periodicboundary = [];
mesh.periodicexpr = {};

% ---------------------------------------------------------
% New Exasim mesh storage format
% ---------------------------------------------------------
mesh.p = mesh.p';
mesh.t = mesh.t';
mesh.f = facenumbering(mesh.p, mesh.t, mesh.elemtype, mesh.boundaryexpr, mesh.periodicexpr);
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

    plot([xwallN xEnd], [ywallN yFlatS], '-r', 'LineWidth', 2);
    plot([0 xFlat0 xCutoff xEnd], [yshift yFlatS yFlatS yFlatS], 'or', 'LineWidth', 2, 'MarkerSize', 7);

    axis equal; axis tight;
    title(sprintf('Rounded-nose flat plate mesh (COARSE, 4x-refinable), Transition at x = %.3f', xCutoff));
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

        % Nose cap
        phi = (pi/2).*xi(inose);
        h   = Hnose.*eta(inose);
        xphys(inose) = rnose - (rnose + h).*cos(phi);
        yphys(inose) =          (rnose + h).*sin(phi) + yshift;

        % Flat body block
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
