% Minimal physics-parameter sweep for the Poisson 2D example.
%
% Each row in pde.physicsparamsweep is one concrete physicsparam vector. The
% generated executable is compiled once, then reused for all cases. Outputs are
% written to dataout/paramcase_0001, dataout/paramcase_0002, ...

run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

[pde,mesh] = initializeexasim();

pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.platform = "cpu";
pde.mpiprocs = 1;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 2;

pde.porder = 2;
pde.pgauss = 2*pde.porder;
pde.physicsparam = 1.0;
pde.physicsparamsweep = [0.5; 1.0; 2.0];
pde.tau = 1.0;
pde.linearsolvertol = 1e-8;
pde.ppdegree = 1;
pde.RBdim = 0;

[mesh.p,mesh.t] = squaremesh(8,8,1,1);
mesh.boundaryexpr = {@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-1)<1e-8, ...
                     @(p) abs(p(2,:)-1)<1e-8, @(p) abs(p(1,:))<1e-8};
mesh.boundarycondition = [1;1;1;1];

[sol,pde,mesh,master,dmd,comstr,runstr] = exasim(pde,mesh);

disp("Sweep outputs:");
for i = 1:numel(pde)
    disp(pde{i}.dataoutpath);
end
