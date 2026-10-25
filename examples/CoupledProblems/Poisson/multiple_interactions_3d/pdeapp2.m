% Domain 2: x in [0.25,0.5].
[pde{2},mesh{2}] = initializeexasim();

% All four domains solve the same Poisson model. The interaction endpoints
% select either column 1 (solution continuity) or column 2 (normal flux) of
% pdemodel.fint.
pde{2}.model = "ModelD";
pde{2}.modelfile = "pdemodel";
pde{2}.Cxxpreprocessing = 1;
pde{2}.platform = "cpu";
pde{2}.mpiprocs = domainranks(2);
pde{2}.hybrid = 1;

% Discretization, physical parameters, and solver controls.
pde{2}.porder = porder;
pde{2}.pgauss = 2*porder;
pde{2}.physicsparam = [1,0];   % unit diffusion; trace interface equation
pde{2}.tau = 1.0;              % HDG stabilization parameter
pde{2}.NLiter = 12;
pde{2}.NLtol = 1e-12;
pde{2}.linearsolvertol = 1e-12;
pde{2}.matvectol = 1e-12;
pde{2}.GMRESrestart = 250;
pde{2}.linearsolveriter = 600;
pde{2}.datapath = runDirectory;
pde{2}.builddir = fullfile(runDirectory,'.exasim');
pde{2}.buildpath = pde{2}.builddir;

% Use N tangential cells on odd domains and 2N on even domains so adjacent
% interface meshes are non-matching.
nt = 3*N;
[mesh{2}.p,mesh{2}.t] = cubemesh(N,nt,nt,families(2));
mesh{2}.p(1,:) = 0.25 + 0.25*mesh{2}.p(1,:);

% Boundary order: y=0, right, y=1, left, z=0, z=1.
mesh{2}.boundaryexpr = {@(p) abs(p(2,:))<1e-8, ...
    @(p) abs(p(1,:)-0.5)<1e-8, @(p) abs(p(2,:)-1)<1e-8, ...
    @(p) abs(p(1,:)-0.25)<1e-8, @(p) abs(p(3,:))<1e-8, ...
    @(p) abs(p(3,:)-1)<1e-8};
mesh{2}.boundarycondition = [1;3;1;2;1;1];

% Generalized interactions are defined in interaction.m, not through the
% legacy mesh.interfacecondition array.
mesh{2}.interfacecondition = [];
