% Domain 3: x in [0.5,0.75].
[pde{3},mesh{3}] = initializeexasim();

% All four domains solve the same Poisson model. The interaction endpoints
% select either column 1 (solution continuity) or column 2 (normal flux) of
% pdemodel.fint.
pde{3}.model = "ModelD";
pde{3}.modelfile = "pdemodel";
pde{3}.Cxxpreprocessing = 1;
pde{3}.platform = "cpu";
pde{3}.mpiprocs = domainranks(3);
pde{3}.hybrid = 1;

% Discretization, physical parameters, and solver controls.
pde{3}.porder = porder;
pde{3}.pgauss = 2*porder;
pde{3}.physicsparam = [1,1];   % unit diffusion; numerical-flux interface equation
pde{3}.tau = 1.0;              % HDG stabilization parameter
pde{3}.NLiter = 12;
pde{3}.NLtol = 1e-12;
pde{3}.linearsolvertol = 1e-12;
pde{3}.matvectol = 1e-12;
pde{3}.GMRESrestart = 250;
pde{3}.linearsolveriter = 600;
pde{3}.datapath = runDirectory;
pde{3}.builddir = fullfile(runDirectory,'.exasim');
pde{3}.buildpath = pde{3}.builddir;

% Use N tangential cells on odd domains and 2N on even domains so adjacent
% interface meshes are non-matching.
nt = N;
[mesh{3}.p,mesh{3}.t] = squaremesh(N,nt,1,families(3));
mesh{3}.p(1,:) = 0.5 + 0.25*mesh{3}.p(1,:);

% Boundary order: y=0, right, y=1, left.
mesh{3}.boundaryexpr = {@(p) abs(p(2,:))<1e-8, ...
    @(p) abs(p(1,:)-0.75)<1e-8, @(p) abs(p(2,:)-1)<1e-8, ...
    @(p) abs(p(1,:)-0.5)<1e-8};
mesh{3}.boundarycondition = [1;3;1;2];

% Generalized interactions are defined in interaction.m, not through the
% legacy mesh.interfacecondition array.
mesh{3}.interfacecondition = [];
