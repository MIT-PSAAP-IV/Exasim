%function [sol,pde,mesh,setupReport,initialReport,finalReport,comparisonReport] = pdeapp(runSolve, runEquilibriumComparison)
%PDEAPP Non-equilibrium five-species-air Mach-8 hypersonic cylinder example.
%
% This case reuses the CNS5air Cartesian compressible Navier--Stokes model
% with finite-rate five-species air chemistry.  The geometry, Mach number,
% wall temperature, length scale, and continuation strategy are chosen to
% match the Mach-8 cylinder examples in examples/NavierStokes as closely as
% possible.
%
% CNS5air conservative-variable ordering in 2-D:
%   [rho_N, rho_O, rho_NO, rho_N2, rho_O2, rho*u, rho*v, rhoE].
%
% Boundary-condition columns from CNS5air/ubound.m and fbouhdgnd.m:
%   1 inflow, 2 outflow, 3 isothermal no-slip wall, 4 slip wall,
%   5 zero-gradient, 6 noncatalytic wall, 7 supercatalytic wall,
%   8 partial-catalysis wall, 9 stoichiometric partial-catalysis wall.

runSolve = 1;
runEquilibriumComparison = 1; 

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
pde.porder = 4;
pde.pgauss = 2*pde.porder;
pde.tau = 8.0;
pde.gencode = 1;
pde.GMRESrestart = 100;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6;
pde.linearsolveriter = 500;
pde.preconditioner = 1;
pde.RBdim = 0;
pde.ppdegree = 20;
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
pde.dt = 1e-3*(1.5.^(0:20));

% Physical conditions matched to the Mach-8 cylinder reference examples.
Minf = 8.03;
ReTarget = 1.835e5;
TinfPhys = 265.0;    % K
TwallPhys = 300.0;   % K
LRef = 1.0;          % m
gammaAir = 1.4;
RAir = 287.05;       % J/(kg K), used only to set the reference Mach speed

eqReference = local_read_equilibrium_reference(repoRoot);
if eqReference.available
    LRef = eqReference.LRef;
    flow = local_freestream_state_from_reference(eqReference.rhoRef, ...
        eqReference.uRef, TinfPhys, LRef);
else
    flow = local_freestream_state(Minf, ReTarget, TinfPhys, LRef, gammaAir, RAir);
end

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

pde.physicsparam = [rhoRef, vRef, rhoeRef, TRef, muRef, kappaRef, ...
                    cpRef, LRef, Ec, Pr, Re, TwallPhys];

Uinf = [flow.rhoSpeciesPhys/rhoRef; ...
        flow.rhovPhys(:)/(rhoRef*vRef); ...
        flow.rhoEPhys/rhoeRef];

Ycat = flow.Y(:);
gammaCatalysis = zeros(5,1);
pde.externalparam = [Uinf; Ycat; gammaCatalysis];

mesh = mkmesh_cyl(pde.porder);
mesh.f = facenumbering(mesh.p,mesh.t,pde.elemtype,mesh.boundaryexpr,mesh.periodicexpr);
% Boundary order from mkmesh_cyl:
%   1 cylinder wall, 2 downstream/outflow, 3 far-field/inflow.
% CNS5air BCs: 6 noncatalytic isothermal wall, 2 outflow, 1 inflow.
mesh.boundarycondition = [6;2;1];

dist = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,[1]);
mesh.dist = dist;
[mesh.udg,mesh.wdg,initializationSource] = local_initial_from_equilibrium_solution( ...
    repoRoot, pde.physicsparam);
if initializationSource.usedEquilibriumSolution == 0
    [mesh.udg,mesh.wdg] = local_initial_state(mesh, dist, flow.Y, rhoRef, vRef, ...
        rhoeRef, TRef, TinfPhys, TwallPhys);
end
mesh.vdg = local_av_field(dist, 0.025, 5);

pde.gencode = 0;
[sol,pde,mesh,master] = exasim(pde,mesh);
sol = sol(:,:,:,end);

% setupReport = local_setup_report(pde, mesh, flow, Minf, ReTarget, Re, Pr, Ec, ...
%     TinfPhys, TwallPhys, LRef);
% setupReport.equilibriumReference = eqReference;
% setupReport.initializationSource = initializationSource;
% initialReport = local_solution_diagnostics(mesh.udg, mesh.wdg, pde.physicsparam);
% local_print_setup(setupReport);
% local_print_diagnostics('initial', initialReport);
% 
% finalReport = local_solution_diagnostics(sol, mesh.wdg, pde.physicsparam);
% local_print_diagnostics('final', finalReport);
% 
% comparisonReport = struct();
% if runEquilibriumComparison
%     comparisonReport = local_compare_equilibrium_case(repoRoot, sol, mesh.wdg, pde.physicsparam);
% end
% 
% save(fullfile(caseDir, 'cylindermach8_report.mat'), ...
%     'setupReport', 'initialReport', 'finalReport', 'comparisonReport');
% 
% try
%     if runSolve == 0
%         return;
%     end
%     figure(1); clf;
%     scaplot(mesh, finalReport.MachField, [0 Minf], 2, 1);
%     axis equal; axis tight; colorbar; colormap('jet');
%     title('Non-equilibrium 5-air Mach number');
%     figure(2); clf;
%     scaplot(mesh, finalReport.TField, [], 2, 1);
%     axis equal; axis tight; colorbar; colormap('jet');
%     title('Non-equilibrium 5-air temperature [K]');
% catch plotError
%     warning('Plotting skipped: %s', plotError.message);
% end
% 
% function flow = local_freestream_state(Minf, ReTarget, Tinf, LRef, gammaAir, RAir)
% velocityPhys = Minf*sqrt(gammaAir*RAir*Tinf);
% rhoPhys = 1.0e-3;
% pPhys = rhoPhys*RAir*Tinf;
% 
% for iter = 1:12
%     info = equilibrate(pPhys, Tinf, [velocityPhys;0]);
%     rhoSpecies = info.rho_species(:);
%     rhoPhys = sum(rhoSpecies);
%     [~,~,~,~,muPhys,kappaPhys,~,cpSpecies,cvSpecies] = ...
%         transportcoefficients(Tinf, rhoSpecies, 1e4);
%     Y = rhoSpecies/rhoPhys;
%     cpMix = sum(double(cpSpecies(:)).*Y);
%     cvMix = sum(double(cvSpecies(:)).*Y);
%     gammaMix = cpMix/cvMix;
%     Rmix = pPhys/(rhoPhys*Tinf);
%     aPhys = sqrt(gammaMix*Rmix*Tinf);
%     velocityPhys = Minf*aPhys;
%     rhoTarget = ReTarget*double(muPhys)/(velocityPhys*LRef);
%     pPhys = rhoTarget*Rmix*Tinf;
%     if abs(rhoTarget-rhoPhys) <= 1e-11*max(1.0, rhoPhys)
%         break;
%     end
% end
% 
% info = equilibrate(pPhys, Tinf, [velocityPhys;0]);
% rhoSpecies = info.rho_species(:);
% rhoPhys = sum(rhoSpecies);
% [rhoEPhys,pCheck,emixPhys] = energyFromSpecies(rhoSpecies, Tinf, [velocityPhys;0], 1e4);
% [~,~,~,~,muPhys,kappaPhys,~,cpSpecies,cvSpecies] = ...
%     transportcoefficients(Tinf, rhoSpecies, 1e4);
% Y = rhoSpecies/rhoPhys;
% cpMix = sum(double(cpSpecies(:)).*Y);
% cvMix = sum(double(cvSpecies(:)).*Y);
% gammaMix = cpMix/cvMix;
% aPhys = velocityPhys/Minf;
% 
% flow = struct();
% flow.rhoSpeciesPhys = rhoSpecies;
% flow.rhoPhys = rhoPhys;
% flow.Y = Y;
% flow.pPhys = pCheck;
% flow.requestedPressure = pPhys;
% flow.TPhys = Tinf;
% flow.velocityPhys = velocityPhys;
% flow.aPhys = aPhys;
% flow.Mach = Minf;
% flow.rhovPhys = rhoPhys*[velocityPhys;0];
% flow.rhoEPhys = rhoEPhys;
% flow.ePhys = emixPhys;
% flow.muPhys = double(muPhys);
% flow.kappaPhys = double(kappaPhys);
% flow.cpMix = cpMix;
% flow.cvMix = cvMix;
% flow.gammaMix = gammaMix;
% flow.Re = rhoPhys*velocityPhys*LRef/flow.muPhys;
% flow.pressureClosureRelativeError = abs(pCheck-pPhys)/max(abs(pCheck),1);
% end
% 
% function flow = local_freestream_state_from_reference(rhoRef, velocityRef, Tinf, LRef)
% % Construct a CNS5air freestream whose physical density and velocity match
% % the already-computed equilibrium-air cylinder reference case.
% pPhys = rhoRef*287.05*Tinf;
% for iter = 1:12
%     info = equilibrate(pPhys, Tinf, [velocityRef;0]);
%     rhoSpecies = info.rho_species(:);
%     rhoPhys = sum(rhoSpecies);
%     pPhys = pPhys * rhoRef/rhoPhys;
%     if abs(rhoPhys-rhoRef) <= 1e-11*max(1.0, rhoRef)
%         break;
%     end
% end
% 
% info = equilibrate(pPhys, Tinf, [velocityRef;0]);
% rhoSpecies = info.rho_species(:);
% rhoPhys = sum(rhoSpecies);
% [rhoEPhys,pCheck,emixPhys] = energyFromSpecies(rhoSpecies, Tinf, [velocityRef;0], 1e4);
% [~,~,~,~,muPhys,kappaPhys,~,cpSpecies,cvSpecies] = ...
%     transportcoefficients(Tinf, rhoSpecies, 1e4);
% Y = rhoSpecies/rhoPhys;
% cpMix = sum(double(cpSpecies(:)).*Y);
% cvMix = sum(double(cvSpecies(:)).*Y);
% gammaMix = cpMix/cvMix;
% aPhys = sqrt(gammaMix*pCheck/rhoPhys);
% 
% flow = struct();
% flow.rhoSpeciesPhys = rhoSpecies;
% flow.rhoPhys = rhoPhys;
% flow.Y = Y;
% flow.pPhys = pCheck;
% flow.requestedPressure = pPhys;
% flow.TPhys = Tinf;
% flow.velocityPhys = velocityRef;
% flow.aPhys = aPhys;
% flow.Mach = velocityRef/aPhys;
% flow.rhovPhys = rhoPhys*[velocityRef;0];
% flow.rhoEPhys = rhoEPhys;
% flow.ePhys = emixPhys;
% flow.muPhys = double(muPhys);
% flow.kappaPhys = double(kappaPhys);
% flow.cpMix = cpMix;
% flow.cvMix = cvMix;
% flow.gammaMix = gammaMix;
% flow.Re = rhoPhys*velocityRef*LRef/flow.muPhys;
% flow.pressureClosureRelativeError = abs(pCheck-pPhys)/max(abs(pCheck),1);
% end
% 
% function eqReference = local_read_equilibrium_reference(repoRoot)
% eqReference = struct();
% eqReference.available = false;
% eqReference.reason = '';
% eqDir = fullfile(repoRoot, 'examples', 'NavierStokes', 'equilibrium5air_cylindermach8');
% appFile = fullfile(eqDir, 'datain', 'app.bin');
% if exist(appFile, 'file') ~= 2
%     eqReference.reason = sprintf('equilibrium app file not found: %s', appFile);
%     return;
% end
% try
%     eqApp = readappbin(appFile);
%     eqMu = eqApp.physicsparam(:);
%     eqReference.available = true;
%     eqReference.appFile = appFile;
%     eqReference.physicsparam = eqMu;
%     eqReference.rhoRef = eqMu(1);
%     eqReference.uRef = eqMu(2);
%     eqReference.pRef = eqMu(3);
%     eqReference.eRef = eqMu(4);
%     eqReference.LRef = eqMu(5);
% catch err
%     eqReference.available = false;
%     eqReference.reason = err.message;
% end
% end
% 
% function [UDG, WDG] = local_initial_state(mesh, dist, Y, rhoRef, vRef, rhoeRef, TRef, Tinf, Twall)
% npe = size(mesh.dgnodes,1);
% ne = size(mesh.dgnodes,3);
% ns = 5;
% UDG = zeros(npe,ns+3,ne);
% WDG = zeros(npe,1,ne);
% 
% speedFactor = tanh(10*dist);
% Tsmooth = Tinf + (Twall - Tinf)*exp(-10*dist);
% rhoSpeciesPhys = rhoRef*Y(:);
% rhoPhys = sum(rhoSpeciesPhys);
% 
% % Use the existing CNS5air thermodynamic routine, but cache it on a modest
% % temperature grid so example startup does not require thousands of NASA
% % polynomial evaluations. This cache affects only the initial guess.
% nCache = 201;
% Tcache = linspace(min(Tsmooth(:)), max(Tsmooth(:)), nCache);
% eCache = zeros(nCache,1);
% for i = 1:nCache
%     [~,~,eCache(i)] = energyFromSpecies(rhoSpeciesPhys, Tcache(i), [0;0], 1e4);
% end
% 
% for ie = 1:ne
%     for ip = 1:npe
%         velocityPhys = [vRef*speedFactor(ip,1,ie); 0];
%         TPhys = Tsmooth(ip,1,ie);
%         emixPhys = interp1(Tcache, eCache, TPhys, 'linear');
%         rhoEPhys = rhoPhys*(emixPhys + 0.5*sum(velocityPhys.^2));
% 
%         UDG(ip,1:ns,ie) = rhoSpeciesPhys(:)'/rhoRef;
%         UDG(ip,ns+1,ie) = sum(rhoSpeciesPhys)*velocityPhys(1)/(rhoRef*vRef);
%         UDG(ip,ns+2,ie) = sum(rhoSpeciesPhys)*velocityPhys(2)/(rhoRef*vRef);
%         UDG(ip,ns+3,ie) = rhoEPhys/rhoeRef;
%         WDG(ip,1,ie) = TPhys/TRef;
%     end
% end
% end
% 
% function VDG = local_av_field(dist, amplitude, slope)
% VDG = zeros(size(dist,1),2,size(dist,3));
% field = amplitude.*tanh(dist*slope);
% VDG(:,1,:) = field;
% VDG(:,2,:) = field;
% end
% 
% function sol = local_stage_solve(pde, mesh, master, dist, sol, avAmplitude, avSlope, label)
% fprintf('%s: artificial-viscosity amplitude %.6g, tanh slope %.6g\n', ...
%     label, avAmplitude, avSlope);
% mesh.vdg = local_av_field(dist, avAmplitude, avSlope);
% mesh.udg = sol;
% [pde,mesh,master,dmd] = preprocessing(pde,mesh); %#ok<ASGLU>
% runcode(pde, 1);
% sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
% sol = local_last_solution(sol);
% local_require_valid_solution(sol, label);
% end
% 
% function sol = local_last_solution(sol)
% if ndims(sol) == 4
%     sol = sol(:,:,:,end);
% end
% end
% 
% function local_require_valid_solution(sol, label)
% if any(~isfinite(sol(:)))
%     error('cylindermach8:%s:NonFiniteSolution', label, ...
%         'The %s returned NaN or Inf values; stopping continuation.', label);
% end
% ns = 5;
% if size(sol,2) >= ns && min(sol(:,1:ns,:),[],'all') <= 0
%     error('cylindermach8:%s:NonPositiveSpecies', label, ...
%         'The %s returned non-positive species density; stopping continuation.', label);
% end
% end
% 
% function setupReport = local_setup_report(pde, mesh, flow, Minf, ReTarget, Re, Pr, Ec, Tinf, Twall, LRef)
% setupReport = struct();
% setupReport.model = 'CNS5air/pdemodel_cart';
% setupReport.stateOrdering = {'rho_N','rho_O','rho_NO','rho_N2','rho_O2','rho_u','rho_v','rhoE'};
% setupReport.speciesOrdering = {'N','O','NO','N2','O2'};
% setupReport.boundaryConditionColumns = mesh.boundarycondition(:).';
% setupReport.boundaryConditionMeaning = {'noncatalytic isothermal wall','outflow','inflow'};
% setupReport.hdg = pde.hybrid == 1;
% setupReport.porder = pde.porder;
% setupReport.pgauss = pde.pgauss;
% setupReport.meshElements = size(mesh.t,2);
% setupReport.meshNodes = size(mesh.p,2);
% setupReport.Minf = Minf;
% setupReport.ReTarget = ReTarget;
% setupReport.ReActual = Re;
% setupReport.Pr = Pr;
% setupReport.Ec = Ec;
% setupReport.TinfPhys = Tinf;
% setupReport.TwallPhys = Twall;
% setupReport.LRef = LRef;
% setupReport.flow = flow;
% setupReport.physicsparam = pde.physicsparam;
% setupReport.externalparam = pde.externalparam;
% end
% 
% function report = local_solution_diagnostics(UDG, WDG, physicsparam)
% ns = 5;
% rhoRef = physicsparam(1);
% vRef = physicsparam(2);
% rhoeRef = physicsparam(3);
% TRef = physicsparam(4);
% LRef = physicsparam(8);
% 
% rhoSpecies = UDG(:,1:ns,:)*rhoRef;
% rho = sum(rhoSpecies,2);
% rhou = UDG(:,ns+1,:)*rhoRef*vRef;
% rhov = UDG(:,ns+2,:)*rhoRef*vRef;
% rhoE = UDG(:,ns+3,:)*rhoeRef;
% ux = rhou./rho;
% uy = rhov./rho;
% speed = sqrt(ux.^2 + uy.^2);
% T = WDG(:,1,:)*TRef;
% e = rhoE./rho - 0.5*(ux.^2 + uy.^2);
% Y = rhoSpecies./rho;
% 
% [~, Mw, RU] = thermodynamicsModels();
% Mw = Mw(:);
% p = T .* sum(rhoSpecies ./ reshape(Mw, [1,ns,1]), 2) * RU;
% omegaMax = 0;
% flatCount = numel(rho);
% sampleCount = min(flatCount, 400);
% sampleIds = unique(round(linspace(1, flatCount, sampleCount)));
% rhoFlat = rho(:);
% TFlat = T(:);
% pFlat = p(:);
% rhoSpeciesFlat = reshape(rhoSpecies, [flatCount, ns]);
% aSample = zeros(numel(sampleIds),1);
% for isample = 1:numel(sampleIds)
%     id = sampleIds(isample);
%     rhoi = rhoSpeciesFlat(id,:).';
%     [~,~,~,~,~,~,~,cpSpecies,cvSpecies] = transportcoefficients(TFlat(id), rhoi, 1e4);
%     Yi = rhoi/sum(rhoi);
%     cpMix = sum(double(cpSpecies(:)).*Yi);
%     cvMix = sum(double(cvSpecies(:)).*Yi);
%     aSample(isample) = sqrt((cpMix/cvMix)*pFlat(id)/rhoFlat(id));
%     omega = kineticsource(TFlat(id), rhoi);
%     omegaMax = max(omegaMax, max(abs(double(omega(:)))));
% end
% aRef = median(aSample);
% Mach = speed./aRef;
% 
% report = struct();
% report.rhoRange = [min(rho(:)), max(rho(:))];
% report.eRange = [min(e(:)), max(e(:))];
% report.pRange = [min(p(:)), max(p(:))];
% report.TRange = [min(T(:)), max(T(:))];
% report.velocityRange = [min(speed(:)), max(speed(:))];
% report.MachRange = [min(Mach(:)), max(Mach(:))];
% report.speciesDensityMin = squeeze(min(min(rhoSpecies,[],1),[],3)).';
% report.speciesDensityMax = squeeze(max(max(rhoSpecies,[],1),[],3)).';
% report.speciesMassFractionMin = squeeze(min(min(Y,[],1),[],3)).';
% report.speciesMassFractionMax = squeeze(max(max(Y,[],1),[],3)).';
% report.YsumErrorMax = max(abs(sum(Y,2)-1),[],'all');
% report.omegaAbsMax = omegaMax;
% report.diagnosticSampleCount = numel(sampleIds);
% report.MachField = Mach;
% report.TField = T;
% report.pField = p;
% report.rhoField = rho;
% report.speedField = speed;
% report.LRef = LRef;
% end
% 
% function local_print_setup(setupReport)
% flow = setupReport.flow;
% fprintf('\nNon-equilibrium five-species-air Mach-8 cylinder setup\n');
% fprintf('  model: %s\n', setupReport.model);
% fprintf('  mesh: %d elements, %d nodes, porder=%d, pgauss=%d, HDG=%d\n', ...
%     setupReport.meshElements, setupReport.meshNodes, setupReport.porder, ...
%     setupReport.pgauss, setupReport.hdg);
% fprintf('  boundary conditions [wall, outflow, inflow] = [%d %d %d]\n', ...
%     setupReport.boundaryConditionColumns);
% fprintf('  freestream: rho=%.8g kg/m^3, T=%.8g K, p=%.8g Pa, |V|=%.8g m/s, a=%.8g m/s, M=%.8g\n', ...
%     flow.rhoPhys, flow.TPhys, flow.pPhys, flow.velocityPhys, flow.aPhys, flow.Mach);
% fprintf('  freestream species Y [N O NO N2 O2] = [%.4e %.4e %.4e %.4e %.4e]\n', flow.Y);
% fprintf('  transport/scales: mu=%.8g Pa s, kappa=%.8g W/(m K), Re=%.8g, Pr=%.8g, Ec=%.8g\n', ...
%     flow.muPhys, flow.kappaPhys, setupReport.ReActual, setupReport.Pr, setupReport.Ec);
% fprintf('  wall: noncatalytic isothermal, Twall=%.8g K\n', setupReport.TwallPhys);
% end
% 
% function local_print_diagnostics(label, report)
% fprintf('\n%s diagnostics\n', label);
% fprintf('  rho range      = [%.8g, %.8g] kg/m^3\n', report.rhoRange);
% fprintf('  e range        = [%.8g, %.8g] J/kg\n', report.eRange);
% fprintf('  p range        = [%.8g, %.8g] Pa\n', report.pRange);
% fprintf('  T range        = [%.8g, %.8g] K\n', report.TRange);
% fprintf('  |V| range      = [%.8g, %.8g] m/s\n', report.velocityRange);
% fprintf('  Mach range     = [%.8g, %.8g]\n', report.MachRange);
% fprintf('  min rho_i      = [%.4e %.4e %.4e %.4e %.4e] kg/m^3\n', report.speciesDensityMin);
% fprintf('  max rho_i      = [%.4e %.4e %.4e %.4e %.4e] kg/m^3\n', report.speciesDensityMax);
% fprintf('  min Y_i        = [%.4e %.4e %.4e %.4e %.4e]\n', report.speciesMassFractionMin);
% fprintf('  max Y_i        = [%.4e %.4e %.4e %.4e %.4e]\n', report.speciesMassFractionMax);
% fprintf('  max |sum(Y)-1| = %.4e\n', report.YsumErrorMax);
% fprintf('  max |omega_i|  = %.4e kg/(m^3 s)\n', report.omegaAbsMax);
% end
% 
% function comparisonReport = local_compare_equilibrium_case(repoRoot, UDG, WDG, physicsparam)
% comparisonReport = struct();
% eqReportFile = fullfile(repoRoot, 'examples', 'NavierStokes', ...
%     'equilibrium5air_cylindermach8', 'equilibrium5air_cylindermach8_report.mat');
% if exist(eqReportFile, 'file') ~= 2
%     fprintf('\nEquilibrium comparison skipped: report file not found at %s\n', eqReportFile);
%     comparisonReport.skipped = true;
%     comparisonReport.reason = 'equilibrium report file not found';
%     return;
% end
% 
% try
%     eqData = load(eqReportFile, 'finalReport');
%     neReport = local_solution_diagnostics(UDG, WDG, physicsparam);
%     comparisonReport.skipped = false;
%     comparisonReport.equilibriumReportFile = eqReportFile;
%     comparisonReport.nonEquilibriumRanges = rmfield(neReport, ...
%         intersect(fieldnames(neReport), {'MachField','TField','pField','rhoField','speedField'}));
%     comparisonReport.equilibriumFinalReport = eqData.finalReport;
%     fprintf('\nEquilibrium comparison loaded from %s\n', eqReportFile);
%     fprintf('  non-equilibrium T range = [%.8g, %.8g] K\n', neReport.TRange);
%     if isfield(eqData.finalReport, 'TRange')
%         fprintf('  equilibrium     T range = [%.8g, %.8g] K\n', eqData.finalReport.TRange);
%     end
% catch err
%     warning('Equilibrium comparison failed: %s', err.message);
%     comparisonReport.skipped = true;
%     comparisonReport.reason = err.message;
% end
% end
