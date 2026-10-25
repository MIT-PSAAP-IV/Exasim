% Domain 1: x in [0,0.25].
[pde{1},mesh{1}] = initializeexasim();

% All four domains solve the same Poisson model. The interaction endpoints
% select either column 1 (solution continuity) or column 2 (normal flux) of
% pdemodel.fint.
pde{1}.model = "ModelD";
pde{1}.modelfile = "pdemodel";
pde{1}.Cxxpreprocessing = 1;
pde{1}.platform = "cpu";
pde{1}.mpiprocs = domainranks(1);
pde{1}.hybrid = 1;

% Discretization, physical parameters, and solver controls.
pde{1}.porder = porder;
pde{1}.pgauss = 2*porder;
pde{1}.physicsparam = [1,1];   % unit diffusion; numerical-flux interface equation
pde{1}.tau = 1.0;              % HDG stabilization parameter
pde{1}.NLiter = 12;
pde{1}.NLtol = 1e-12;
pde{1}.linearsolvertol = 1e-12;
pde{1}.matvectol = 1e-12;
pde{1}.GMRESrestart = 250;
pde{1}.linearsolveriter = 600;
pde{1}.datapath = runDirectory;
pde{1}.builddir = fullfile(runDirectory,'.exasim');
pde{1}.buildpath = pde{1}.builddir;

% Use 4N tangential cells on odd domains and 3N on even domains so adjacent
% interface meshes are non-matching.
nt = 4*N;
[mesh{1}.p,mesh{1}.t] = squaremesh(N,nt,1,families(1));
mesh{1}.p(1,:) = 0.25*mesh{1}.p(1,:);

% Boundary order: y=0, right, y=1, left.
mesh{1}.boundaryexpr = {@(p) abs(p(2,:))<1e-8, ...
    @(p) abs(p(1,:)-0.25)<1e-8, @(p) abs(p(2,:)-1)<1e-8, ...
    @(p) abs(p(1,:))<1e-8};
mesh{1}.boundarycondition = [1;3;1;1];

% Generalized interactions are defined in interaction.m, not through the
% legacy mesh.interfacecondition array.
mesh{1}.interfacecondition = [];
