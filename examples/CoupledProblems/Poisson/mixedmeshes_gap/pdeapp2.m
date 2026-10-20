% Model 2: [0.5+gap/2,1] x [0,1].
[pde{2},mesh{2}] = initializeexasim();

pde{2}.model = "ModelD";
pde{2}.modelfile = "pdemodel2";
pde{2}.modelnumber = 2;
pde{2}.Cxxpreprocessing = 1;
pde{2}.mpiprocs = 1;
pde{2}.hybrid = 1;
pde{2}.porder = 3;
pde{2}.physicsparam = 1;
pde{2}.tau = 1.0;
pde{2}.NLiter = 3;
pde{2}.NLtol = 1e-10;
pde{2}.linearsolvertol = 1e-10;
pde{2}.matvectol = 1e-10;
pde{2}.GMRESrestart = 300;
pde{2}.linearsolveriter = 1000;
pde{2}.interfacetype = 1;

if ~exist('triangleOnModel1', 'var'), triangleOnModel1 = true; end
if ~exist('gap', 'var'), gap = 0.02; end
elemtype = double(triangleOnModel1);
[mesh{2}.p,mesh{2}.t] = squaremesh(12,24,1,elemtype);
interface = 0.5+gap/2;
mesh{2}.p(1,:) = (1-interface)*mesh{2}.p(1,:) + interface;
mesh{2}.boundaryexpr = {@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-1)<1e-8, ...
    @(p) abs(p(2,:)-1)<1e-8, @(p) abs(p(1,:)-interface)<1e-8};
mesh{2}.boundarycondition = [1;1;1;2];
mesh{2}.interfacecondition = [0;0;0;1];
pde{2}.interfacefluxmap = 1;
