% Domain 2: quadrilateral mesh on x in [0.5,1] by default.
[pde{2},mesh{2}] = initializeexasim();

pde{2}.model = "ModelD";
pde{2}.modelfile = "pdemodel2";
pde{2}.modelnumber = 2;
pde{2}.Cxxpreprocessing = 1;
pde{2}.platform = "cpu";
pde{2}.mpiprocs = domainranks(2);
pde{2}.hybrid = 1;

pde{2}.porder = 3;
pde{2}.pgauss = 2*pde{2}.porder;
pde{2}.physicsparam = 1;
pde{2}.tau = 1.0;
pde{2}.NLiter = 3;
pde{2}.NLtol = 1e-10;
pde{2}.linearsolvertol = 1e-10;
pde{2}.matvectol = 1e-10;
pde{2}.GMRESrestart = 300;
pde{2}.linearsolveriter = 1000;
pde{2}.datapath = runDirectory;
pde{2}.builddir = fullfile(runDirectory,'.exasim');
pde{2}.buildpath = pde{2}.builddir;

elemtype = double(triangleOnModel1);
[mesh{2}.p,mesh{2}.t] = squaremesh(12,24,1,elemtype);
mesh{2}.p(1,:) = 0.5*mesh{2}.p(1,:) + 0.5;
mesh{2}.boundaryexpr = {@(p) abs(p(2,:))<1e-8, ...
    @(p) abs(p(1,:)-1)<1e-8, @(p) abs(p(2,:)-1)<1e-8, ...
    @(p) abs(p(1,:)-0.5)<1e-8};
mesh{2}.boundarycondition = [1;1;1;2];

% interaction.m selects the left boundary; the legacy interface array is off.
mesh{2}.interfacecondition = [];
