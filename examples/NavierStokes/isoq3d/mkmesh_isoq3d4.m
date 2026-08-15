function mesh = mkmesh_isoq3d4(porder, res, nz)
%MKMESH_ISOQ3D4  v4 "true-geometry" isoq3d fluid mesh generator.
%
%   mesh = mkmesh_isoq3d4(porder, res, nz)
%
%   Reconstruction of the missing v4 generator that produced the committed
%   mesh_fluid/*_truegeom meshes.  Differs from the v3 pipeline
%   (mkmesh_isoq2d2 -> mkmesh_isoq3d3) as follows:
%     * NO 0.002 m radial standoff: the wall sits exactly on the isoq profile.
%     * The rotation axis (r=0) is filled by a butterfly / O-grid plug
%       (spherical-frustum core built from mkmesh_quartercircle), so there is
%       no collapsed-node singularity and no inner-cylinder boundary.
%     * 5 boundaries [4,4,2,3,1] (v3 had 6, [4,4,2,4,3,1]) -- boundary #4, the
%       inner-cylinder standoff surface, is gone.
%
%   res = [n1 m1 n2 m2] : surface/normal subdivisions of the two shell blocks.
%   nz                  : azimuthal element layers over the quarter [0,pi/2]
%                         (must be even). Also sets the plug arc resolution.
%
%   Element count model (validated against the committed meshes):
%     ne = (n1*m1 + n2*m2)*nz  +  3*(nz/2)^2 * m1
%          \______ shell ______/    \___ butterfly plug ___/
%   The shell term reproduces the "legacy" rotated-mesh cost exactly
%   (coarse 4352, medium 26880, fine 107520). The plug is ~1-3% of the mesh
%   and removes the axis singularity; the committed meshes vary in the plug
%   block count (medium 1 block, coarse/fine 2 blocks) -- here it is a
%   conforming 3-block quarter butterfly (nq = nz).

if nargin < 1 || isempty(porder); porder = 2; end
if nargin < 2 || isempty(res);    res = [48 80 36 80]; end
if nargin < 3 || isempty(nz);     nz  = 4; end
if mod(nz,2) ~= 0; error('mkmesh_isoq3d4: nz must be even (butterfly plug needs nz/2).'); end

% ---- graded body-fitted shell (wall on profile, dR = 0) ----
[mesh2d, xc, xd] = mkmesh_isoq2d4(porder, res);
[p, t, xdg] = rotate_mesh(mesh2d.p, mesh2d.t, nz, mesh2d.dgnodes);

mesh = mkmesh(p', t', porder, {'true'}, 1, 1);
mesh.p = mesh.p'; mesh.t = mesh.t';
mesh.dgnodes = xdg;

% ---- axis butterfly plug (spherical-frustum core, arc resolution = nz) ----
xc = [xc; 0*xc(1,:)];
xd = [xd; 0*xd(1,:)];
[xl, ~, Rn] = isoq();
plug = mkmesh_sphericalfrustum(xc, porder, nz, Rn, xd);

% ---- stitch plug + shell ----
[p, t] = connectmesh(plug.p', plug.t', mesh.p', mesh.t');
xdg_all = cat(3, plug.dgnodes, xdg);
mesh = mkmesh(p, t, porder, {'true'}, 1, 1);
mesh.p = mesh.p'; mesh.t = mesh.t';
mesh.dgnodes = xdg_all;

% ---- 5 truegeom boundaries, in the committed order ----
ymin = min(mesh.p(2,:)); zmin = min(mesh.p(3,:)); xmax = max(mesh.p(1,:));
tol = 1e-6;
mesh.boundaryexpr = { ...
    @(p) abs(p(2,:)-ymin) < tol, ...                                     % 1 y=0 symmetry
    @(p) abs(p(3,:)-zmin) < tol, ...                                     % 2 z=0 symmetry
    @(p) abs(p(1,:)-xmax) < tol, ...                                     % 3 x=xE outflow
    @(p) -1e-3<p(1,:) & p(1,:)<xmax+1e-3 & sqrt(p(2,:).^2+p(3,:).^2)<0.06, ... % 4 wall
    @(p) abs(p(1,:)) < 20 + 1e-6};                                       % 5 far field
mesh.boundarycondition = [4, 4, 2, 3, 1];
mesh.f = facenumbering(mesh.p, mesh.t, 1, mesh.boundaryexpr, []);
mesh.periodicboundary = [];
mesh.periodicexpr = {};

% ---- wall distance field (vdg / odg), wall = boundary 4 ----
mesh.dist = meshdist3(mesh.f, mesh.dgnodes, mesh.perm, 4);
mesh.xmax = xmax; mesh.ymin = ymin; mesh.zmin = zmin;
end
