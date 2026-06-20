function mesh = pdeapp_el(mesh, pde, bump_amp, bump_loc, bump_width)
% Linear elasticity mesh deformation for wall bump

pde_el = pde;
pde_el.modelnumber = 2;   % isolate from NS model (model 0) and HM model (model 1)
E = 1.0;
nu = 0.30;
mu_lam = E / (2 * (1 + nu));
lambda = nu * E / ((1 + nu) * (1 - 2 * nu));

pde_el.model = "ModelD";
pde_el.modelfile = "pdemodel_elastic";
pde_el.physicsparam = [mu_lam, lambda, bump_amp, bump_loc, bump_width];
pde_el.tau = 2*(mu_lam + lambda);
pde_el.linearsolvertol = 1e-8;
pde_el.GMRESrestart = 100;
pde_el.linearsolveriter = 200;
pde_el.NLtol = 1e-8;
pde_el.NLiter = 1;
pde_el.gencode = 1;

mesh_el = mesh;
mesh_el.boundarycondition = [1, 1, 2, 1];

master_el = Master(pde_el);
[sol_el, ~, ~, ~, ~] = exasim(pde_el, mesh_el);
figure();scaplot(mesh, -sol_el(:,2,:));

mesh.dgnodes = mesh.dgnodes - sol_el(:, 1:2, :);
figure(1);clf;meshplot(mesh);

end
