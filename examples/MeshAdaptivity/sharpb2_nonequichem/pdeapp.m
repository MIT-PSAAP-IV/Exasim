% Sharp-B axisymmetric finite-rate five-species-air mesh-adaptivity example.

caseDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(fileparts(caseDir)));
run(fullfile(repoRoot,'frontends','Matlab','exasim_setup.m'));
addpath(caseDir,'-begin');
addpath(fullfile(repoRoot,'frontends','Matlab','Modeling','CNS5air'),'-begin');
% Reuse the Sharp-B mesh construction without duplicating it.
addpath(fullfile(repoRoot,'examples','MeshAdaptivity', ...
    'sharpb2_idealgas'),'-end');

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
pde.RBdim = 10;
pde.ppdegree = 0;
pde.NLtol = 1e-6;
pde.NLiter = 5;
pde.matvectol = 1e-6;
pde.dae_alpha = 0;
pde.dae_beta = 0;
pde.dae_gamma = 0;
pde.dt = [1e-5 1e-4 1e-3 4e-3 2e-2 1e-1];
pde.nstage = 1;
pde.torder = 1;
pde.saveSolFreq = length(pde.dt);
pde.saveParaview = 1;
pde.datapath = caseDir;
pde.builddir = fullfile(caseDir,'.exasim');
pde.buildpath = pde.builddir;

% Match the Mach, Reynolds number, and temperatures of sharpb2_equichem.
Minf = 21.38;
ReTarget = 9.84e5;
TinfPhysical = 260.6;
TwallPhysical = 1400.0;
lengthReference = 1.0;

flow = local_freestream_state(Minf,ReTarget,TinfPhysical,lengthReference);
pressurePhysical = flow.pressure;
velocityPhysical = flow.velocity;
rhoSpeciesPhysical = flow.rhoSpecies;
rhoPhysical = flow.rho;
rhovPhysical = rhoPhysical*velocityPhysical;
rhoEPhysical = flow.rhoE;

rhoReference = rhoPhysical;
velocityReference = velocityPhysical;
rhoeReference = rhoReference*velocityReference^2;
temperatureReference = TinfPhysical;
viscosityReference = flow.mu;
conductivityReference = flow.kappa;
cpReference = flow.cp;

Re = rhoReference*lengthReference*velocityReference/viscosityReference;
Pr = viscosityReference*cpReference/conductivityReference;
Ec = velocityReference^2/(cpReference*temperatureReference);

nm = 1e2;
pde.AV = 2;
pde.AVcontinuationIter = 10;
pde.AVcontinuationLogScale = 2.0;
pde.AVcoeffStart = 0.005;
pde.AVcoeffEnd = 0.00005;
pde.AVdistfunction = 1;
pde.distanceboundaryconditions = 8; % noncatalytic isothermal-wall BC tag
pde.AVsmoothingMethod = 1;
pde.AVHelmholtzCoeff = 0.001;
AVmaxdiv = 20.0;
AVdistcoeff = nm;

speciesDensityMinimum = -1e-5;
densityMinimum = -1e-5;
temperatureMinimum = 100.0;
temperatureMaximum = 2.0e4;
pressureMinimum = 1.0e-8*pressurePhysical;

pde.meshadaptenabled = 1;
pde.meshadaptfield = 2; % physical pressure from avfield
pde.meshadaptavcomponent = 1;
pde.meshadaptalpha = 0.1;
pde.meshadaptqmin = 0.2;
pde.meshadaptqmax = 0.8;
pde.meshadaptHelmholtzCoeff = 0.001;
pde.meshadaptforcescale = 0.7;
pde.meshadaptsmoothingpasses = 30;
% Geometric boundaries: axis, lower farfield, upper farfield, wall, outflow.
pde.meshadaptboundaryconditions = [3;3;3;2;3];

pde.physicsparam = [rhoReference,velocityReference,rhoeReference, ...
    temperatureReference,viscosityReference,conductivityReference, ...
    cpReference,lengthReference,Ec,Pr,Re,TwallPhysical, ...
    speciesDensityMinimum,densityMinimum,temperatureMinimum, ...
    temperatureMaximum,pressureMinimum, ...
    AVmaxdiv,AVdistcoeff,pde.AVcoeffStart,pde.AVcoeffEnd];

Uinf = [rhoSpeciesPhysical(:)/rhoReference; ...
        rhovPhysical/(rhoReference*velocityReference); ...
        0; rhoEPhysical/rhoeReference];
[rhoEWallPhysical,~] = energyFromSpecies( ...
    rhoSpeciesPhysical,TwallPhysical,[0;0],1.0e4);
wallInternalEnergy = rhoEWallPhysical/rhoeReference;
Ycat = rhoSpeciesPhysical(:)/rhoPhysical;
gammaCatalysis = zeros(5,1);
pde.externalparam = [Uinf;Ycat;gammaCatalysis];

mesh = mkmesh_sharpb2(pde.porder);
% Symmetry, inflow, inflow, noncatalytic isothermal wall, outflow.
mesh.boundarycondition = [5 1 1 8 2];
master = Master(pde);
dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,4);

mesh.vdg = zeros(size(mesh.dgnodes,1),3,size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = dist;
[mesh.udg,mesh.wdg] = local_initial_solution( ...
    mesh,dist,Uinf,wallInternalEnergy,TwallPhysical/TinfPhysical,nm);

soundSpeedInf = soundspeed(TinfPhysical,rhoSpeciesPhysical);
fprintf('\nFinite-rate five-species-air Sharp-B setup\n');
fprintf('  rho_inf = %.10g kg/m^3, p_inf = %.10g Pa, T_inf = %.10g K\n', ...
    rhoPhysical,pressurePhysical,TinfPhysical);
fprintf('  a_inf = %.10g m/s, U_inf = %.10g m/s, Mach = %.8g, Re = %.8g\n', ...
    soundSpeedInf,velocityPhysical,velocityPhysical/soundSpeedInf,Re);

pde.gencode = 1;
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);
sol = sol(:,:,:,end);

xdg = getsolution(fullfile(caseDir,'dataout','outxdg'),dmd,master.npe);
vdg = getsolution(fullfile(caseDir,'dataout','outvdg'),dmd,master.npe);
wdg = getsolutions(fullfile(caseDir,'dataout','outwdg'),dmd);
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
figure(2); clf; scaplot(mesh,mach,[],2,2);
axis equal; axis tight; colorbar;
figure(3); clf; scaplot(mesh,pde.avparam1(end)+pde.avparam2(end)*vdg(:,2,:),[],2,2);
axis equal; axis tight; colorbar;
figure(4); clf; scaplot(mesh,pressurePhysical,[],2);
axis equal; axis tight; colorbar;
figure(5); clf; scaplot(mesh,rhoReference*rho,[],2);
axis equal; axis tight; colorbar;
figure(6); clf; scaplot(mesh,wdg,[],2);
axis equal; axis tight; colorbar;


function [UDG,WDG] = local_initial_solution( ...
    mesh,dist,Uinf,wallInternalEnergy,wallTemperature,slope)
rho = sum(Uinf(1:5));
uz = (Uinf(6)/rho)*tanh(slope*dist);
ur = (Uinf(7)/rho)*tanh(slope*dist);
wallWeight = exp(-slope*dist);
freestreamInternalEnergy = Uinf(8)-0.5*(Uinf(6)^2+Uinf(7)^2)/rho;
internalEnergyDensity = freestreamInternalEnergy + ...
    (wallInternalEnergy-freestreamInternalEnergy).*wallWeight;

UDG = zeros(size(mesh.dgnodes,1),8,size(mesh.dgnodes,3));
for species = 1:5
    UDG(:,species,:) = Uinf(species);
end
UDG(:,6,:) = rho.*uz;
UDG(:,7,:) = rho.*ur;
UDG(:,8,:) = internalEnergyDensity+0.5*rho.*(uz.^2+ur.^2);
WDG = 1.0+(wallTemperature-1.0).*wallWeight;
end

function flow = local_freestream_state(Mach,Reynolds,Tinf,lengthReference)
% Use the established Sharp-B freestream density to set pressure. Reference
% transport scales are arbitrary when Re, Pr, and Ec are formed from the
% same scales, so this avoids symbolic transport evaluation at startup.
rhoSeed = 0.00235;
pressure = 287.0*rhoSeed*Tinf;
info = equilibrate(pressure,Tinf,0.0);
rhoSpecies = info.rho_species;
rho = info.rho;
[soundSpeed,~,cp] = soundspeed(Tinf,rhoSpecies);
velocity = Mach*soundSpeed;
[rhoE,pressure] = energyFromSpecies(rhoSpecies,Tinf,velocity,1.0e4);
mu = rho*lengthReference*velocity/Reynolds;
kappa = mu*cp/0.71;
flow = struct('pressure',pressure,'velocity',velocity, ...
    'rhoSpecies',rhoSpecies,'rho',rho,'rhoE',rhoE, ...
    'mu',mu,'kappa',kappa,'cp',cp);
end
