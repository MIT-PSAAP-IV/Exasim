% Domain 1: triangular mesh on x in [0,0.5] by default.
[pde{1},mesh{1}] = initializeexasim();

pde{1}.model = "ModelD";
pde{1}.modelfile = "pdemodel1";
pde{1}.modelnumber = 1;
pde{1}.Cxxpreprocessing = 1;
pde{1}.platform = "cpu";
pde{1}.mpiprocs = domainranks(1);
pde{1}.hybrid = 1;

pde{1}.porder = 3;
pde{1}.pgauss = 2*pde{1}.porder;
pde{1}.physicsparam = 1;
pde{1}.tau = 1.0;
pde{1}.NLiter = 3;
pde{1}.NLtol = 1e-10;
pde{1}.linearsolvertol = 1e-10;
pde{1}.matvectol = 1e-10;
pde{1}.GMRESrestart = 300;
pde{1}.linearsolveriter = 1000;
pde{1}.datapath = runDirectory;
pde{1}.builddir = fullfile(runDirectory,'.exasim');
pde{1}.buildpath = pde{1}.builddir;

elemtype = double(~triangleOnModel1);
[mesh{1}.p,mesh{1}.t] = squaremesh(8,16,1,elemtype);
mesh{1}.p(1,:) = 0.5*mesh{1}.p(1,:);
mesh{1}.boundaryexpr = {@(p) abs(p(2,:))<1e-8, ...
    @(p) abs(p(1,:)-0.5)<1e-8, @(p) abs(p(2,:)-1)<1e-8, ...
    @(p) abs(p(1,:))<1e-8};
mesh{1}.boundarycondition = [1;2;1;1];

% interaction.m selects the right boundary; the legacy interface array is off.
mesh{1}.interfacecondition = [];
