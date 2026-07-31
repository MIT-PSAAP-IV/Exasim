% Poisson verification problem for mkmesh_auplate2d.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

[pde,mesh] = initializeexasim();

pde.model = "ModelD";
pde.modelfile = "pdemodel_poi";

pde.platform = "cpu";
pde.mpiprocs = 4;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 2;

pde.porder = 2;
pde.pgauss = 2*pde.porder;
pde.physicsparam = 1.0;
pde.tau = 1.0;
pde.linearsolvertol = 1.0e-10;
pde.linearsolveriter = 200;
pde.GMRESrestart = 200;
pde.preconditioner = 1;
pde.ppdegree = 0;
pde.RBdim = 0;
pde.saveParaview = 1;

mesh = mkmesh_auplate2d(pde.porder);

% mkmesh_auplate2d boundary numbering:
%   1 symmetry centerline        -> homogeneous Neumann
%   2 inflow / outer nose arc    -> Dirichlet u = 1
%   3 rounded nose wall          -> Dirichlet u = 0
%   4 flat plate wall            -> Dirichlet u = 0
%   5 outflow                    -> homogeneous Neumann
%   6 freestream top             -> Dirichlet u = 1
mesh.boundarycondition = [2 1 3 3 2 1];

figure(1); clf;
for ib = 1:numel(mesh.boundarycondition)
    boundaryplot(mesh, ib); hold on;
end
axis equal;
axis tight;
title('auplate Poisson mesh boundary groups');

[sol,pde,mesh,master,dmd,comstr,runstr] = exasim(pde,mesh);

mesh.porder = pde.porder;
figure(2); clf;
scaplot(mesh, sol(:,1,:), [], 2);
axis on;
axis equal;
axis tight;
colorbar;
title('Poisson solution on auplate mesh');
