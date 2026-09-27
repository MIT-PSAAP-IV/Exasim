caseDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(fileparts(caseDir)));
run(fullfile(repoRoot, 'frontends', 'Matlab', 'exasim_setup.m'));
validPrefix = fullfile(fileparts(repoRoot), 'exasim_install');
if exist(fullfile(validPrefix, 'lib', 'cmake', 'Exasim', 'ExasimConfig.cmake'), 'file') == 2
    setenv('EXASIM_PREFIX', validPrefix);
end
addpath(caseDir, '-begin');
addpath(fullfile(repoRoot, 'frontends', 'Matlab', 'Modeling', 'CNS5air'), '-begin');
addpath(fullfile(repoRoot, 'frontends', 'Matlab', 'Modeling', 'CNSequilibrium5air'), '-begin');
addpath(fullfile(repoRoot, 'frontends', 'Matlab', 'Materials'), '-begin');

[pde,~] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel_cart";
pde.platform = "cpu";
pde.mpiprocs = 8;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 2;
pde.elemtype = 1;
pde.porder = 2;
pde.pgauss = 2*pde.porder;
pde.tau = 8.0;
pde.gencode = 1;
pde.GMRESrestart = 500;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6;
pde.linearsolveriter = 500;
pde.preconditioner = 1;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;
pde.NLiter = 10;
pde.matvectol = 1e-6;
pde.dae_alpha = 0;
pde.dae_beta = 0;
pde.dae_gamma = 0;
pde.saveSolFreq = 1;
pde.datapath = caseDir;
pde.builddir = fullfile(caseDir, '.exasim');
pde.buildpath = pde.builddir;

pde.dt = [0.1 1 10];

% Physical conditions matched to the Mach-8 cylinder reference examples.
Minf = 8.03;
ReTarget = 1.835e5;
TinfPhys = 265.0;    % K
TwallPhys = 300.0;   % K
LRef = 1.0;          % m
gammaAir = 1.4;
RAir = 287.05;       % J/(kg K), used only to set the reference Mach speed

flow = local_freestream_state(Minf, ReTarget, TinfPhys, LRef, gammaAir, RAir);
rhoRef = flow.rhoPhys;
vRef = flow.velocityPhys;
TRef = TinfPhys;
rhoeRef = rhoRef*vRef^2;
pRef = rhoeRef;
muRef = flow.muPhys;
kappaRef = flow.kappaPhys;
cpRef = flow.cpMix;
Pr = muRef*cpRef/kappaRef;
Re = rhoRef*LRef*vRef/muRef;
Ec = vRef^2/(cpRef*TRef);

pde.AV = 1;
pde.AVcontinuationIter = 10;
pde.AVcontinuationLogScale = 1.0;
pde.AVcoeffStart = 0.060;
pde.AVcoeffEnd = 0.015;
pde.AVdistfunction = 1;
pde.distanceboundaryconditions = [6];
pde.AVsmoothingMethod = 1;
pde.AVHelmholtzCoeff = 0.025;
AVmaxdiv = 2.0;
AVdistcoeff = 20;

pde.meshadaptenabled = 1;
pde.meshadaptfield = 1;
pde.meshadaptalpha = 0.5;
pde.meshadaptHelmholtzCoeff = 5e-2;
pde.meshadaptforcescale = 0.2;
pde.meshadaptsmoothingpasses = 30;
pde.meshadaptboundaryconditions = [2;3;3];

pde.physicsparam = [rhoRef, vRef, rhoeRef, TRef, muRef, kappaRef, ...
                    cpRef, LRef, Ec, Pr, Re, TwallPhys, ...
                    AVmaxdiv, AVdistcoeff, pde.AVcoeffStart, pde.AVcoeffEnd];

Uinf = [flow.rhoSpeciesPhys/rhoRef; ...
        flow.rhovPhys(:)/(rhoRef*vRef); ...
        flow.rhoEPhys/rhoeRef];

Ycat = flow.Y(:);
gammaCatalysis = zeros(5,1);
pde.externalparam = [Uinf; Ycat; gammaCatalysis];

mesh = mkmesh_square(51,32,pde.porder,1,1,1,1,1);
mesh.p(1,:) = logdec(mesh.p(1,:), 3);
mesh.dgnodes(:,1,:) = logdec(mesh.dgnodes(:,1,:), 3);
mesh = mkmesh_halfcircle(mesh, 1, 3, 4.0, pi/2, 3*pi/2);
mesh.porder = pde.porder;
mesh.boundaryexpr = {@(p) sqrt(p(1,:).^2+p(2,:).^2)<1+1e-6, ...
                     @(p) p(1,:)>-1e-7, @(p) abs(p(1,:))<20};
mesh.periodicexpr = {};

mesh.f = facenumbering(mesh.p,mesh.t,pde.elemtype,mesh.boundaryexpr,mesh.periodicexpr);
% CNS5air BCs: 6 noncatalytic isothermal wall, 2 outflow, 1 inflow.
mesh.boundarycondition = [6;2;1];

dist = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,[1]);

mesh.dist = dist;
mesh.vdg = zeros(size(mesh.dgnodes,1),2,size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = dist;

rho = sum(Uinf(1:5));
ux = (Uinf(6)/rho)*tanh(10*dist);
uy = (Uinf(7)/rho)*tanh(10*dist);
internalEnergy = Uinf(8) - 0.5*(Uinf(6)^2 + Uinf(7)^2)/rho;
mesh.udg = zeros(size(mesh.dgnodes,1),8,size(mesh.dgnodes,3));
for species = 1:5
    mesh.udg(:,species,:) = Uinf(species);
end
mesh.udg(:,6,:) = rho.*ux;
mesh.udg(:,7,:) = rho.*uy;
mesh.udg(:,8,:) = internalEnergy + 0.5*rho.*(ux.^2 + uy.^2);
mesh.wdg = ones(size(mesh.dgnodes,1),1,size(mesh.dgnodes,3));

pde.gencode = 1;
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);
sol = sol(:,:,:,end);
vdg = getsolution('dataout/outvdg', dmd, master.npe);
xdg = getsolution('dataout/outxdg', dmd, master.npe);
wdg = getsolutions('dataout/outwdg', dmd);
wdg = wdg(:,:,:,end);
mesh.dgnodes = xdg;

rho = sum(sol(:,1:5,:),2);
figure(1); clf; scaplot(mesh, rhoRef*rho,[],2,2);
axis equal; axis tight; colorbar; colormap jet;

figure(2); clf; scaplot(mesh, vRef*sol(:,6,:)./rho,[],2,2);
axis equal; axis tight; colorbar; colormap jet;

figure(3); clf; scaplot(mesh, vRef*sol(:,7,:)./rho,[],2,2);
axis equal; axis tight; colorbar; colormap jet;

Tphys = TRef .* wdg(:,1,:);
figure(4); clf; scaplot(mesh, Tphys,[],2,2);
axis equal; axis tight; colorbar; colormap jet;

[~, Mw, RU] = thermodynamicsModels();
rhoSpeciesPhys = rhoRef .* sol(:,1:5,:);
rhom = sum(rhoSpeciesPhys ./ reshape(Mw,1,5,1), 2);
Pphys = RU .* Tphys .* rhom;
figure(5); clf; scaplot(mesh, Pphys,[],2,2);
axis equal; axis tight; colorbar; colormap jet;

[a, gammaMix, cpMix, cvMix, Rmix] = soundspeed(Tphys, abs(rhoSpeciesPhys));

vx = sol(:,6,:)./rho;
vy = sol(:,7,:)./rho;
vv = vRef*sqrt(vx.^2 + vy.^2);
Mach = vv./a;
figure(6); clf; scaplot(mesh, Mach,[],2,2);
axis equal; axis tight; colorbar; colormap jet;

figure(7); clf; scaplot(mesh, vdg(:,2,:),[],2,2);
axis equal; axis tight; colorbar;

figure(8); clf; meshplot(mesh,1);
axis equal; axis tight; colorbar;

