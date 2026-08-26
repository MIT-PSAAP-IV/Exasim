% Export the true-geometry isoq3d application (mkmesh_isoq3d4: wall on the
% analytic isoq profile, butterfly axis plug, NO radial standoff) as a
% Text2Code app.
%
% Differences from pdeapp3_exporttext2code.m:
%   - mesh built by mkmesh_isoq3d4 (5 boundaries; the inner standoff
%     cylinder no longer exists), boundarycondition = [4, 4, 2, 3, 1]
%   - initial condition interpolated from the saved 2D solution by (x, r)
%     with scatteredInterpolant instead of element-wise extrusion (the plug
%     elements have no 2D parent element). The 2D source coordinates are
%     UNSHIFTED by dR first so its wall coincides with the true profile.
%
% After exporting, update the CHEFSI runtime inputs (input_exasim_*.txt):
%   boundaryconditions = [4, 4, 2, 3, 1];
% The wall keeps FbouHdg block 3, so the coupled driver's ibc = 2 is
% unchanged. Copy grid.bin/xdg.bin/udg.bin/vdg.bin into
% CHEFSI-apps/FSP-1/ExaSim-SumMIT/isoq3d/mesh_fluid/<coarse|fine>/.

run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

% Initial-condition SOURCE: the previously exported (shifted-mesh) 3D app
% bins. solp2coarse.mat no longer exists, but the exported udg/vdg ARE the
% extruded 2D solution, so collapsing them back to (x, r) recovers the same
% axisymmetric source data.
if ~exist('src_name', 'var'), src_name = 'isoq3d'; end   % 'isoq3d_finemesh' for the fine source
srcdir = fullfile(fileparts(mfilename('fullpath')), ...
                  '..', '..', '..', 'apps', 'navierstokes', src_name);
xdgS = readbin3(fullfile(srcdir, 'xdg.bin'));   % npe x 3 x ne
udgS = readbin3(fullfile(srcdir, 'udg.bin'));   % npe x 5 x ne (rho,ru,rv,rw,rE)
vdgS = readbin3(fullfile(srcdir, 'vdg.bin'));   % npe x 1 x ne (AV)

[pde,~] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel";

pde.platform = "cpu";
pde.mpiprocs = 8;
pde.porder = 2;
pde.pgauss = 2*pde.porder;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 3;

gam = 1.4;
Re = 1.835e5;
Pr = 0.71;
Minf = 7;
Tref  = 124.49;
Twall = 294.44;
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
alpha = 0;
rinf = 1.0;
ruinf = cos(alpha);
rvinf = sin(alpha);
rwinf = 0;
rEinf = 0.5+pinf/(gam-1);

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rwinf rEinf Tinf Tref Twall];
pde.tau = 10.0;
pde.GMRESrestart = 200;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6;
pde.linearsolveriter = 200;
pde.preconditioner = 1;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;
pde.NLiter = 10;
pde.matvectol = 1e-6;

pde.dt = 1e-4*(1.2.^(0:7));
pde.nstage = 1;
pde.torder = 1;
pde.saveSolFreq = 4;

% ---------------------------------------------------------------- new mesh
% Optional base-workspace overrides (for -batch driving):
%   mesh_res    : [n1 m1 n2 m2] block resolution (default: generator default)
%   mesh_npeel  : axis peel layers (default 2; use 1 for the old-cost mesh)
%   export_name : apps/navierstokes/<name> export directory
% Known recipes:
%   old-cost coarse : mesh_res=[20 32 14 32],   mesh_npeel=1, mesh_nz=4
%   medium (default): mesh_res=[48 80 36 80],   mesh_npeel=2, mesh_nz=4
%   fine (old cost) : mesh_res=[64 120 48 120], mesh_npeel=2, mesh_nz=8,
%                     src_name='isoq3d_finemesh'
if ~exist('mesh_res', 'var'),    mesh_res = [48 80 36 80]; end
if ~exist('mesh_npeel', 'var'),  mesh_npeel = 2; end
if ~exist('mesh_nz', 'var'),     mesh_nz = 4; end
if ~exist('export_name', 'var'), export_name = 'isoq3d_truegeom'; end
nz = mesh_nz;
mesh = mkmesh_isoq3d4(pde.porder, mesh_npeel, nz, mesh_res);

ymin = min(mesh.p(2,:));
zmin = min(mesh.p(3,:));
xmax = max(mesh.p(1,:));
tol = 1e-8;
mesh.boundaryexpr = [ ...
    "abs(y-(" + num2str(ymin, 17) + "))<" + num2str(tol, 17), ...
    "abs(z-(" + num2str(zmin, 17) + "))<" + num2str(tol, 17), ...
    "abs(x-(" + num2str(xmax, 17) + "))<" + num2str(tol, 17), ...
    "(-1e-3<x)&&(x<(" + num2str(xmax, 17) + ")+1e-3)&&(sqrt(y^2+z^2)<0.06)", ...
    "abs(x)<20+1e-6"];
mesh.boundarycondition = [4, 4, 2, 3, 1];

% -------------------------------------- initial condition by (x, r) lookup
% Collapse the (SHIFTED) 3D source to axisymmetric (x, r) samples: unshift
% the radius by the standoff dR, convert (ruy, ruz) to radial momentum.
XS = reshape(xdgS(:,1,:), [], 1);
YS = reshape(xdgS(:,2,:), [], 1);
ZS = reshape(xdgS(:,3,:), [], 1);
RS = sqrt(YS.^2 + ZS.^2);
dR = min(RS);                                % standoff of the old mesh
fprintf('[ic] source standoff dR = %.6f m\n', dR);
rsrc = RS - dR;                              % unshifted radius
rhoS = reshape(udgS(:,1,:), [], 1);
ruS  = reshape(udgS(:,2,:), [], 1);
ryS  = reshape(udgS(:,3,:), [], 1);
rzS  = reshape(udgS(:,4,:), [], 1);
rES  = reshape(udgS(:,5,:), [], 1);
rvS  = (YS.*ryS + ZS.*rzS) ./ max(RS, 1e-30);   % radial momentum
avS  = reshape(vdgS(:,1,:), [], 1);

% dedupe identical (x, r) rows (azimuthal copies collapse onto each other)
[~, iu] = unique(round([XS rsrc]/1e-9)*1e-9, 'rows');
xsrc = XS(iu); rsrc = rsrc(iu);
Frho = scatteredInterpolant(xsrc, rsrc, rhoS(iu), 'linear', 'nearest');
Fru  = scatteredInterpolant(xsrc, rsrc, ruS(iu),  'linear', 'nearest');
Frv  = scatteredInterpolant(xsrc, rsrc, rvS(iu),  'linear', 'nearest');
FrE  = scatteredInterpolant(xsrc, rsrc, rES(iu),  'linear', 'nearest');
Fav  = scatteredInterpolant(xsrc, rsrc, avS(iu),  'linear', 'nearest');

[npe, ~, ne] = size(mesh.dgnodes);
X = reshape(mesh.dgnodes(:,1,:), [], 1);
Y = reshape(mesh.dgnodes(:,2,:), [], 1);
Z = reshape(mesh.dgnodes(:,3,:), [], 1);
R = sqrt(Y.^2 + Z.^2);
TH = atan2(Z, max(Y, 0) + (Y <= 0).*Y);     % atan2(z, y)

rho = Frho(X, R);
ru  = Fru(X, R);
rv  = Frv(X, R);                            % radial momentum in the section
rE  = FrE(X, R);
av  = Fav(X, R);

% On the axis (R -> 0) the radial momentum must vanish; the interpolant
% may return a small finite value from the source's dR standoff line.
taper = min(R / (4*dR), 1.0);
rv = rv .* taper;

mesh.udg = zeros(npe, 5, ne);
mesh.udg(:,1,:) = reshape(rho, npe, 1, ne);
mesh.udg(:,2,:) = reshape(ru,  npe, 1, ne);
mesh.udg(:,3,:) = reshape(rv .* cos(TH), npe, 1, ne);
mesh.udg(:,4,:) = reshape(rv .* sin(TH), npe, 1, ne);
mesh.udg(:,5,:) = reshape(rE,  npe, 1, ne);
mesh.vdg = reshape(av, npe, 1, ne);

% ------------------------------------------------------------------ export
exasimroot = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
exportdir = fullfile(exasimroot, 'apps', 'navierstokes', export_name);
exporttext2code(pde, mesh, exportdir);

function a = readbin3(fn)
fid = fopen(fn, 'r');
if fid < 0, error('cannot open %s', fn); end
h = fread(fid, 3, 'double');
a = reshape(fread(fid, prod(h), 'double'), h(1), h(2), h(3));
fclose(fid);
end
