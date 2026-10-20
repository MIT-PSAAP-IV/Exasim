% Model 1: [0,0.5] x [0,1].
[pde{1},mesh{1}] = initializeexasim();

pde{1}.model = "ModelD";
pde{1}.modelfile = "pdemodel1";
pde{1}.modelnumber = 1;
pde{1}.Cxxpreprocessing = 1;
pde{1}.mpiprocs = 1;
pde{1}.hybrid = 1;
pde{1}.porder = 3;
pde{1}.physicsparam = 1;
pde{1}.tau = 1.0;
pde{1}.NLiter = 3;
pde{1}.NLtol = 1e-10;
pde{1}.linearsolvertol = 1e-10;
pde{1}.matvectol = 1e-10;
pde{1}.GMRESrestart = 300;
pde{1}.linearsolveriter = 1000;
pde{1}.interfacetype = 1;

if ~exist('triangleOnModel1', 'var'), triangleOnModel1 = true; end
if ~exist('gap', 'var'), gap = 0.02; end
elemtype = double(~triangleOnModel1);
[mesh{1}.p,mesh{1}.t] = squaremesh(8,16,1,elemtype);
interface = 0.5-gap/2;
mesh{1}.p(1,:) = interface*mesh{1}.p(1,:);
mesh{1}.boundaryexpr = {@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-interface)<1e-8, ...
    @(p) abs(p(2,:)-1)<1e-8, @(p) abs(p(1,:))<1e-8};
mesh{1}.boundarycondition = [1;2;1;1];
mesh{1}.interfacecondition = [0;1;0;0];
pde{1}.interfacefluxmap = 1;
