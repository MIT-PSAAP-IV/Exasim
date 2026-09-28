% Axisymmetric finite-rate five-species-air ISOQ mesh-adaptivity example.

caseDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(fileparts(caseDir)));
%run(fullfile(repoRoot,'frontends','Matlab','exasim_setup.m'));
addpath(caseDir,'-begin');
addpath(fullfile(repoRoot,'frontends','Matlab','Modeling','CNS5air'),'-begin');
addpath(fullfile(repoRoot,'examples','MeshAdaptivity','isoq2d_idealgas'),'-end');

[pde,~] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.platform = "cpu";
pde.mpiprocs = 8;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 2;
pde.elemtype = 1;
pde.porder = 2;
pde.pgauss = 2*pde.porder;
pde.tau = 10.0;
pde.GMRESrestart = 250;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6;
pde.linearsolveriter = 250;
pde.preconditioner = 1;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;
pde.NLiter = 5;
pde.matvectol = 1e-6;
pde.dae_alpha = 0;
pde.dae_beta = 0;
pde.dae_gamma = 0;
pde.dt = [0.1 1]/10;
pde.nstage = 1;
pde.torder = 1;
pde.saveSolFreq = 1;
pde.saveParaview = 1;
pde.datapath = caseDir;
pde.builddir = fullfile(caseDir,'.exasim');
pde.buildpath = pde.builddir;

% Freestream and reference state from the reacting ISOQ case.
TinfPhysical = 266.5;
TwallPhysical = 300.0;
rhoPhysicalGuess = 1.047e-3;
velocityPhysical = 2500.0;
pressurePhysical = 288.0*rhoPhysicalGuess*TinfPhysical;
lengthReference = 1.0;

[rhoSpeciesPhysical,rhoPhysical,rhovPhysical,rhoEPhysical] = ...
    getEquilibriumState(pressurePhysical,TinfPhysical,velocityPhysical);
soundSpeedInf = soundspeed(TinfPhysical,rhoSpeciesPhysical);
Minf = velocityPhysical/soundSpeedInf;
[rhoReference,velocityReference,rhoeReference,~,temperatureReference, ...
    viscosityReference,conductivityReference,~,cpReference] = ...
    getReferenceState(pressurePhysical,TinfPhysical,velocityPhysical);

Re = rhoReference*lengthReference*velocityReference/viscosityReference;
Pr = viscosityReference*cpReference/conductivityReference;
Ec = velocityReference^2/(cpReference*temperatureReference);

nm = 1e2;
pde.AV = 1;
pde.AVcontinuationIter = 10;
pde.AVcontinuationLogScale = 3.0;
pde.AVcoeffStart = 2e-3;
pde.AVcoeffEnd = 2e-5;
pde.AVdistfunction = 1;
pde.distanceboundaryconditions = 8; % 8 catalytic isothermal wall
pde.AVsmoothingMethod = 1;
pde.AVHelmholtzCoeff = 0.001;
AVmaxdiv = 30.0;
AVdistcoeff = nm;
speciesDensityMinimum = -1e-5;
densityMinimum = -1e-5;
temperatureMinimum = 100.0;
temperatureMaximum = 2.0e4;
pressureMinimum = 1.0e-8*pressurePhysical;

pde.meshadaptenabled = 1;
pde.meshadaptfield = 1; % physical pressure from visscalars
pde.meshadaptavcomponent = 1;
pde.meshadaptalpha = 0.5;
pde.meshadaptqmin = 0.2;
pde.meshadaptqmax = 0.8;
pde.meshadaptHelmholtzCoeff = 0.001;
pde.meshadaptforcescale = 0.1;
pde.meshadaptsmoothingpasses = 30;
pde.meshadaptboundaryconditions = [3;3;3;2];

pde.physicsparam = [rhoReference,velocityReference,rhoeReference, ...
    temperatureReference,viscosityReference,conductivityReference, ...
    cpReference,lengthReference,Ec,Pr,Re,TwallPhysical, ...
    speciesDensityMinimum,densityMinimum,temperatureMinimum, ...
    temperatureMaximum,pressureMinimum, ...
    AVmaxdiv,AVdistcoeff,pde.AVcoeffStart,pde.AVcoeffEnd];

Uinf = [rhoSpeciesPhysical(:)/rhoReference; ...
        rhovPhysical(:)/(rhoReference*velocityReference); ...
        0; rhoEPhysical/rhoeReference];
Ycat = rhoSpeciesPhysical(:)/rhoPhysical;
gammaCatalysis = zeros(5,1);
pde.externalparam = [Uinf;Ycat;gammaCatalysis];

mesh = mkmesh_isoq2d(pde.porder,5e-4);
% Slip/axis symmetry, outflow, inflow, catalytic isothermal wall.
mesh.boundarycondition = [5 2 1 8];
master = Master(pde);
dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,4);

mesh.vdg = zeros(size(mesh.dgnodes,1),2,size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = dist;
mesh.udg = local_initial_solution(mesh,dist,Uinf,nm);
mesh.wdg = ones(size(mesh.dgnodes,1),1,size(mesh.dgnodes,3));

fprintf('\nFinite-rate five-species-air ISOQ setup\n');
fprintf('  rho_inf = %.10g kg/m^3, T_inf = %.10g K, U_inf = %.10g m/s\n', ...
    rhoPhysical,TinfPhysical,velocityPhysical);
fprintf('  Mach = %.8g, Re = %.8g\n',Minf,Re);

pde.gencode = 1;
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);
sol = sol(:,:,:,end);

xdg = getsolution('dataout/outxdg',dmd,master.npe);
vdg = getsolution('dataout/outvdg',dmd,master.npe);
wdg = getsolutions('dataout/outwdg',dmd);
wdg = wdg(:,:,:,end);
mesh.dgnodes = xdg;

rho = sum(sol(:,1:5,:),2);
rhoSpeciesPhysical = rhoReference*sol(:,1:5,:);
temperaturePhysical = temperatureReference*wdg(:,1,:);
[~,Mw,RU] = thermodynamicsModels();
molarDensity = sum(rhoSpeciesPhysical./reshape(Mw,1,5,1),2);
pressurePhysical = RU*temperaturePhysical.*molarDensity;
soundSpeed = soundspeed(temperaturePhysical,abs(rhoSpeciesPhysical));
velocity = velocityReference*sol(:,6:7,:)./rho;
mach = sqrt(sum(velocity.^2,2))./soundSpeed;

figure(1); clf; meshplot(mesh,1); axis equal; axis tight;
figure(2); clf; scaplot(mesh,mach,[0 Minf],2,1); axis equal; axis tight; colorbar;
figure(3); clf; scaplot(mesh,pde.avparam1(end) + pde.avparam2(end)*vdg(:,2,:),[],2,2); axis equal; axis tight; colorbar;
figure(4); clf; scaplot(mesh,pressurePhysical,[],2,2); axis equal; axis tight; colorbar;

function UDG = local_initial_solution(mesh,dist,Uinf,slope)
rho = sum(Uinf(1:5));
uz = (Uinf(6)/rho)*tanh(slope*dist);
ur = (Uinf(7)/rho)*tanh(slope*dist);
internalEnergyDensity = Uinf(8)-0.5*(Uinf(6)^2+Uinf(7)^2)/rho;

UDG = zeros(size(mesh.dgnodes,1),8,size(mesh.dgnodes,3));
for species = 1:5
    UDG(:,species,:) = Uinf(species);
end
UDG(:,6,:) = rho.*uz;
UDG(:,7,:) = rho.*ur;
UDG(:,8,:) = internalEnergyDensity+0.5*rho.*(uz.^2+ur.^2);
end
