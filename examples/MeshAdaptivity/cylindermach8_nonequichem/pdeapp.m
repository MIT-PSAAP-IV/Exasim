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
% CNS5air BCs: 6 noncatalytic isothermal wall, 2 outflow, 1 inflow.
mesh.boundarycondition = [6;2;1];

dist = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,[1]);
[mesh.udg,mesh.wdg] = local_initial_from_equilibrium_solution(repoRoot, pde.physicsparam);
mesh.vdg = 0.025*tanh(dist*5);
mesh.vdg(:,2,:) = 0.025*tanh(dist*5);

pde.gencode = 1;
[sol,pde,mesh,master] = exasim(pde,mesh);
sol = sol(:,:,:,end);

