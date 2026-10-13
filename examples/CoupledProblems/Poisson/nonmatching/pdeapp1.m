% Model 1 of the non-matching coupled Poisson example: [0,0.5] x [0,1].
[pde{1},mesh{1}] = initializeexasim();

pde{1}.model = "ModelD";
pde{1}.modelfile = "pdemodel1";
pde{1}.modelnumber = 1;
pde{1}.Cxxpreprocessing = 1;
pde{1}.mpiprocs = 1;
pde{1}.hybrid = 1;
pde{1}.porder = 2;
pde{1}.physicsparam = 1;
pde{1}.tau = 1.0;
pde{1}.NLiter = 3;
pde{1}.NLtol = 1e-10;
pde{1}.linearsolvertol = 1e-10;
pde{1}.matvectol = 1e-10;
pde{1}.GMRESrestart = 300;
pde{1}.linearsolveriter = 1000;

% Non-matching interface (app.problem[34] = 1): the backend locates donor points on
% the other model's independently generated mesh; no frontend matching is needed.
pde{1}.interfacetype = 1;

% 8 x 16 quadrilaterals: interface faces of length 1/16.
[mesh{1}.p,mesh{1}.t] = squaremesh(8,16,1,1);
mesh{1}.p(1,:) = 0.5*mesh{1}.p(1,:);
mesh{1}.boundaryexpr = {@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-0.5)<1e-8, @(p) abs(p(2,:)-1)<1e-8, @(p) abs(p(1,:))<1e-8};
mesh{1}.boundarycondition = [1;2;1;1];
mesh{1}.interfacecondition = [0;1;0;0];   % boundary 2 is the coupled interface
pde{1}.interfacefluxmap = 1;
