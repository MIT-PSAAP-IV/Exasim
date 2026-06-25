porder = 4;                    % polynomial degree
gam = 1.4;                     % gas constant
Minf = 0.1;                    % freestream mach number
tau = 12;                      % stabilization parameter
rinf = 1.0;                    % freestream density
pinf = 1/(gam*Minf^2);         % freestream pressure
rEinf = 0.5+pinf/(gam-1);      % freestream energy
Pr = 0.72;                     % Prandtl number
% reynoldsNumbers = [2000; 5000; 8000]; % [1000, 10000]
% anglesOfAttack = [4; 8]*pi/180;       % [0, 10]

% reynoldsNumbers = [2000; 5000; 8000]; % [1000, 10000]
% anglesOfAttack = [4; 8]*pi/180;       % [0, 10]

alpha = 3;
n = 17;
reynoldsNumbers = logdec(linspace(1000,10000,n), alpha);
anglesOfAttack  = logdec(linspace(0,10*pi/180,n), alpha);

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();

pde.model = "ModelD";          % ModelC, ModelD, ModelW
pde.modelfile = "pdemodel";    % name of a file defining the PDE model

% Choose computing platform and set number of processors
pde.platform = "cpu";          % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 8;              % number of MPI processors
pde.hybrid = 1;
pde.debugmode = 0;
pde.porder = porder;
pde.pgauss = 2*porder;

basePhysicsParam = [gam reynoldsNumbers(1) Pr Minf rinf cos(anglesOfAttack(1)) sin(anglesOfAttack(1)) rEinf];
pde.physicsparam = basePhysicsParam;
pde.physicsparamsweep = zeros(numel(reynoldsNumbers)*numel(anglesOfAttack), numel(basePhysicsParam));
icase = 0;
for ialpha = 1:numel(anglesOfAttack)
    alpha = anglesOfAttack(ialpha);
    for iRe = 1:numel(reynoldsNumbers)
        icase = icase + 1;
        pde.physicsparamsweep(icase,:) = [gam reynoldsNumbers(iRe) Pr Minf rinf cos(alpha) sin(alpha) rEinf];
    end
end

% 
tm = pde.physicsparamsweep;
%pde.physicsparamsweep = tm(1:42,:);
%pde.physicsparamsweep = tm(43:84,:);
%pde.physicsparamsweep = tm(85:(85+41),:);
%pde.physicsparamsweep = tm(127:(126+41),:);
%pde.physicsparamsweep = tm(168:(167+41),:);
%pde.physicsparamsweep = tm(209:(208+41),:);
pde.physicsparamsweep = tm(250:289,:);

pde.tau = tau;                 % DG stabilization parameter
pde.GMRESrestart = 100;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6;    % GMRES tolerance
pde.linearsolveriter = 100;
pde.preconditioner = 1;
pde.NLtol = 1e-8;
pde.ppdegree = 0;
pde.RBdim = 5;
pde.gencode = 1;

pde.torder = 3;
pde.nstage = 3;
pde.dt = 0.05*ones(1,400);
pde.saveSolFreq = 2;
pde.saveSolBouFreq = 2;
pde.ibs = 1;

% Export a frontend-provider app that can run the entire sweep without MATLAB.
pde.exportapp = "naca-sweep7";
pde.frontendprovider = true;
pde.buildandrun = false;
if exist(pde.exportapp, 'dir')
    rmdir(pde.exportapp, 's');
end

% naca mesh
mesh = mkmesh_naca0012(porder,1,3);

% call exasim to preprocess, generate code, and export the standalone app
exasim(pde,mesh);

fprintf("Exported NACA0012 sweep app: %s\n", fullfile(pwd, pde.exportapp));
fprintf("Run with: EXASIM_ROOT=%s %s\n", char(exasim_install_prefix()), fullfile(pwd, pde.exportapp, "run.sh"));

