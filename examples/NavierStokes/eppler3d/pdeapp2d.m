% Add Exasim to Matlab search path
cdir = pwd(); ii = strfind(cdir, "Exasim");
run(cdir(1:(ii+5)) + "/install/setpath.m");

porder = 2;                     % polynomial degree
gam = 1.4;                      % gas constant
Minf = 0.09;                    % freestream Mach number
tau = 10;                        % stabilization parameter
alpha = 4*pi/180;               % angle of attack
beta = 0;                       % sideslip angle
rinf = 1.0;                     % freestream density
ruinf = cos(alpha)*cos(beta);   % freestream x momentum
rvinf = sin(alpha);             % freestream y momentum
pinf = 1/(gam*Minf^2);          % freestream pressure
rEinf = 0.5 + pinf/(gam-1);     % freestream energy
Re = 300000;                     % Reynolds number
Pr = 0.72;                      % Prandtl number

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();

pde.model = "ModelD";
pde.modelfile = "pdemodel2d";

% Choose computing platform and set number of processors
pde.platform = "cpu";
pde.mpiprocs = 4;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 2;
pde.porder = porder;
pde.pgauss = 2*porder;

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf];
pde.tau = tau;
pde.GMRESrestart = 60;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-7;
pde.linearsolveriter = 60;
pde.preconditioner = 1;
pde.NLtol = 1e-7;
pde.NLiter = 3;
pde.ppdegree = 0;
pde.RBdim = 5;
pde.gencode = 1;

pde.torder = 3;
pde.nstage = 3;
pde.dt = [1e-4 2e-4 4e-4 8e-4 1.6e-3 3.2e-3 0.005*ones(1,4000)];
pde.saveSolFreq = 100;

% Spanwise extrusion of the Eppler 387 C-grid.
% mesh = mkmesh_epp387(porder, 1, -2);

TEC = 15;
sps = [TEC, 1, 1, 1, 1, TEC, 1, 1, 1, 1, TEC];
spr = [10, 10, 10, 10, 10, 10, 10]*70;
yref = [0.0025 0.008 0.02 0.036];
lw = 10;
ll = 10;
nxw = 21;
nflr = 11;
nflf = 11;
nfuf = 15;
nfur = 21;
nr   = 41;
mesh = clemesh_airfoil(xf, yf, nxw, nflr, nflf, nfuf, nfur, nr, sps, spr, yref, lw, ll, porder);

[pde,mesh,master,dmd] = preprocessing(pde,mesh);
UDG = getsolution("eppler2d/dataout/paramcase_0004/outudg_t5000",dmd,9);
figure(1); clf; scaplot(mesh, UDG(:,2,:)./UDG(:,1,:),[],2); colormap('jet'); colorbar;

% call exasim to generate and run C++ code to solve the PDE model
% pde.exportapp = "eppler2d";
% pde.frontendprovider = true;
% pde.buildandrun = false;
% [sol,pde,mesh,master,dmd] = exasim(pde,mesh);

%UDG1 = dgprojection(master1,mesh1,UDG,porder)
% UDG = getsolution("eppler2d/dataout/outudg_t4000",dmd,9);
% figure(1); clf; scaplot(mesh, UDG(:,2,:)./UDG(:,1,:),[],2); colormap('jet'); colorbar;
% figure(2); clf; scaplot(mesh, eulereval(UDG,'r',1.4,Minf),[],2); colormap('jet'); colorbar;

% % plot final density field
% figure(1); clf;
% scaplot3d(mesh, sol(:,1,:,end), [], 1);
% colorbar; axis equal; axis tight;
