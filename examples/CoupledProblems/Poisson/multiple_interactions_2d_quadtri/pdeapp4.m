% Domain 4: x in [0.75,1].
[pde{4},mesh{4}] = initializeexasim();

% All four domains solve the same Poisson model. The interaction endpoints
% select either column 1 (solution continuity) or column 2 (normal flux) of
% pdemodel.fint.
pde{4}.model = "ModelD";
pde{4}.modelfile = "pdemodel";
pde{4}.Cxxpreprocessing = 1;
pde{4}.platform = "cpu";
pde{4}.mpiprocs = domainranks(4);
pde{4}.hybrid = 1;

% Discretization, physical parameters, and solver controls.
pde{4}.porder = porder;
pde{4}.pgauss = 2*porder;
pde{4}.physicsparam = [1,0];   % unit diffusion; trace interface equation
pde{4}.tau = 1.0;              % HDG stabilization parameter
pde{4}.NLiter = 12;
pde{4}.NLtol = 1e-12;
pde{4}.linearsolvertol = 1e-12;
pde{4}.matvectol = 1e-12;
pde{4}.GMRESrestart = 250;
pde{4}.linearsolveriter = 600;
pde{4}.datapath = runDirectory;
pde{4}.builddir = fullfile(runDirectory,'.exasim');
pde{4}.buildpath = pde{4}.builddir;

% Use N tangential cells on odd domains and 2N on even domains so adjacent
% interface meshes are non-matching.
nt = 3*N;
[mesh{4}.p,mesh{4}.t] = squaremesh(N,nt,1,families(4));
mesh{4}.p(1,:) = 0.75 + 0.25*mesh{4}.p(1,:);

% Boundary order: y=0, right, y=1, left.
mesh{4}.boundaryexpr = {@(p) abs(p(2,:))<1e-8, ...
    @(p) abs(p(1,:)-1)<1e-8, @(p) abs(p(2,:)-1)<1e-8, ...
    @(p) abs(p(1,:)-0.75)<1e-8};
mesh{4}.boundarycondition = [1;1;1;2];

% Generalized interactions are defined in interaction.m, not through the
% legacy mesh.interfacecondition array.
mesh{4}.interfacecondition = [];
