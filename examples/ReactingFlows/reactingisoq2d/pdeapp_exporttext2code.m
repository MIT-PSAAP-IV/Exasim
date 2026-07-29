% Export the isoq3d MATLAB frontend application as a Text2Code app.
%
% This script mirrors pdeapp3.m through mesh/state construction, then writes a
% self-contained Text2Code package under apps/navierstokes/isoq3d.

run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

load('solp2.mat');
mesh2d = mesh; udg2d = mesh.udg(:,1:4,:); vdg2d = mesh.vdg;

[pde,~] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel";

pde.platform = "cpu";
pde.mpiprocs = 16;
pde.porder = 2;
pde.pgauss = 2*pde.porder;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 3;

gam = 1.4;                      % specific heat ratio
Re = 1.56e5;                    % Reynolds number
Pr = 0.71;                      % Prandtl number
Minf = 7.6;                     % Mach number
Tref  = 266.5;
Twall = 300;
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
alpha = 0;                      % angle of attack
rinf = 1.0;                     % freestream density
ruinf = cos(alpha);             % freestream horizontal velocity
rvinf = sin(alpha);             % freestream vertical velocity
rwinf = 0;
rEinf = 0.5+pinf/(gam-1);       % freestream energy

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rwinf rEinf Tinf Tref Twall];
pde.tau = 10.0;
pde.GMRESrestart = 200;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6;
pde.linearsolveriter = 400;
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

nz = 8;
mesh = mkmesh_isoq3d3(mesh2d, nz);
mesh.boundarycondition = [4, 4, 2, 4, 3, 1];

r2d = mesh2d.dgnodes(:,2,:);
dR = min(r2d(:));
ymin = min(mesh.p(2,:));
zmin = min(mesh.p(3,:));
xmax = max(mesh.p(1,:));
tol = 1e-8;
mesh.boundaryexpr = [ ...
    "abs(y-(" + num2str(ymin, 17) + "))<" + num2str(tol, 17), ...
    "abs(z-(" + num2str(zmin, 17) + "))<" + num2str(tol, 17), ...
    "abs(x-(" + num2str(xmax, 17) + "))<" + num2str(tol, 17), ...
    "abs(y^2+z^2-(" + num2str(dR, 17) + ")^2)<" + num2str(tol, 17), ...
    "(-1e-3<x)&&(x<(" + num2str(xmax, 17) + ")+1e-3)&&(sqrt(y^2+z^2)<0.06+(" + num2str(dR, 17) + "))", ...
    "abs(x)<20+1e-6"];

mesh.vdg = extrudesol(vdg2d, pde.porder, nz);

tt = linspace(0, pi/2, nz+1);
[vx3d, vy3d] = extrudevelocity(udg2d(:,3,:), pde.porder, tt);
udg3d = extrudesol(udg2d, pde.porder, nz);
mesh.udg = udg3d(:,1:2,:);
mesh.udg(:,3,:) = vx3d;
mesh.udg(:,4,:) = vy3d;
mesh.udg(:,5,:) = udg3d(:,4,:);

exasimroot = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
exportdir = fullfile(exasimroot, 'apps', 'navierstokes', 'isoq3d');
exporttext2code(pde, mesh, exportdir);
