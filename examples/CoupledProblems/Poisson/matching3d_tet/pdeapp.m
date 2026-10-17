% 3D coupled Poisson problem on two subdomains of the unit cube with a
% matching interface x = 0.5 (tetrahedra, interfacetype = 0). The exact
% solution is u = exp(x) sin(pi y) sin(pi z); model 1 covers x <= 0.5 and
% model 2 x >= 0.5. Both meshes come from cubemesh.

clear pde mesh master dmd sol;
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

pdeapp1;
pdeapp2;

[sol,pde,mesh,master,dmd] = exasim(pde,mesh);

% Maximum nodal error against the exact solution. A one-rank model returns its
% solution in partition order, so it is put back into global element order first.
err = 0;
for m = 1:2
    elemtype = double(size(mesh{m}.t, 1) == 8);
    f = facenumbering(mesh{m}.p, mesh{m}.t, elemtype, mesh{m}.boundaryexpr, []);
    xdg = createdgnodes(mesh{m}.p, mesh{m}.t, f, [], [], pde{m}.porder);
    u = sol{m};
    if numel(dmd{m}) == 1
        u(:,:,dmd{m}{1}.elempart(1:size(u,3))) = sol{m};
    end
    e = u(:,1,:) - exp(xdg(:,1,:)).*sin(pi*xdg(:,2,:)).*sin(pi*xdg(:,3,:));
    err = max(err, max(abs(e(:))));
end
fprintf('Maximum nodal error against exp(x) sin(pi y) sin(pi z): %.3e\n', err);
