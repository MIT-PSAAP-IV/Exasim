cdir = pwd(); ii = strfind(cdir, "Exasim");
run(cdir(1:(ii+5)) + "/install/setpath.m");
addpath(cdir(1:(ii+5)) + "/backend/Model/BuiltIn");

% Build path with modelnumber suffix
buildpath = pwd;
modelnumber = 1;
datainpath = buildpath + "/datain" + num2str(modelnumber);
dataoutpath = buildpath + "/dataout" + num2str(modelnumber);
if ~exist(datainpath, 'dir'), mkdir(datainpath); end
if ~exist(dataoutpath, 'dir'), mkdir(dataoutpath); end

% Read mesh from grid.bin
[mesh.p, mesh.t] = readmesh("grid.bin", 0);

% Boundary expressions matching the triangular mesh:
mesh.boundaryexpr = {@(p) abs(p(2,:)-1e-4)<1e-6, ...
                     @(p) abs(p(1,:)-0.3492)<1e-6, ...
                     @(p) abs(p(2,:) - (1e-4 + 0.538*(p(1,:)+0.1101)))<2e-3};
mesh.boundarycondition = [1; 1; 1];
mesh.curvedboundary = [];
mesh.curvedboundaryexpr = {};
mesh.periodicboundary = [];
mesh.periodicexpr = {};

% Initialize pde structure for model6
pde = initializeexasim();
pde.builtinmodelID = 6;
pde.modelfile = "pdemodel6";
pde.model = "ModelD";
pde.hybrid = 1;
pde.modelnumber = modelnumber;
pde.buildpath = buildpath;
pde.datapath = buildpath;
pde.datainpath = datainpath;
pde.dataoutpath = dataoutpath;
pde.exasimpath = cdir(1:(ii+5));
pde.porder = 2;
pde.pgauss = 4;
pde.torder = 1;
pde.nstage = 1;
pde.ncu = 2;
pde.ncw = 0;
pde.ncv = 0;
pde.ntau = 1;
pde.tau = [1];
pde.physicsparam = [1; 1];
pde.NLtol = 1e-8;
pde.NLiter = 10;
pde.linearsolveriter = 200;
pde.GMREStol = 1e-8;
pde.GMRESrestart = 50;
pde.precMatrixType = 0;
pde.preconditioner = 0;
pde.meshfile = "n";
pde.xdgfile = "n";
pde.udgfile = "n";
pde.mpiprocs = 1;
pde.platform = "cpu";
pde.debugmode = 0;
pde.time = 0;
pde.dt = [0];
pde.saveSolFreq = 0;
pde.gendatain = 1;
pde.preprocessmode = 1;

% Run preprocessing to generate datain/ files
[app, mesh, master, dmd] = preprocessing(pde, mesh);

fprintf("Done preprocessing mesh6.\n");
