% Coupled Poisson problem on two independently meshed subdomains with a
% non-matching interface (interfacetype = 1). The exact solution is
% u = sin(pi x) sin(pi y) on [0,1]^2; model 1 covers x <= 0.5, model 2 x >= 0.5.
%
% Set curved = true to bend the interface to x = 0.5 + 0.05 sin(pi y): both sides
% then interpolate the curve at their own face nodes, so the two discrete interfaces
% leave small gaps and overlaps that the non-matching coupling bridges.

clear pde mesh master dmd sol;
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

curved = false;

pdeapp1;
pdeapp2;

% DG nodes in the global element order of each mesh, optionally mapped to the
% curved interface (the map fixes the outer boundaries). preprocessing() uses
% mesh{m}.dgnodes when it is present.
for m = 1:2
    mesh{m}.dgnodes = dgnodes_of(mesh{m}, pde{m}.porder, m, curved);
end

% The solver prints "Non-matching donor search, Gamma^1" and "..., Gamma^2"
% diagnostics (faces, points, maximum point-to-face distance, points off their
% donor face) for both coupling directions during setup.
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);

% Put the solution of a one-rank model into the global element order as well
% (several ranks are already assembled in that order), then compare with the
% exact solution at the DG nodes.
err = 0;
for m = 1:2
    mesh{m}.porder = pde{m}.porder;
    mesh{m}.dgnodes = dgnodes_of(mesh{m}, pde{m}.porder, m, curved);
    if numel(dmd{m}) == 1
        u = sol{m};
        u(:,:,dmd{m}{1}.elempart(1:size(u,3))) = sol{m};
        sol{m} = u;
    end
    x = mesh{m}.dgnodes;
    e = sol{m}(:,1,:) - sin(pi*x(:,1,:)).*sin(pi*x(:,2,:));
    err = max(err, max(abs(e(:))));
end
fprintf('Maximum nodal error against sin(pi x) sin(pi y): %.3e\n', err);

figure(1); clf;
scaplot(mesh{1}, sol{1}(:,1,:), [], 2, 1); hold on;
scaplot(mesh{2}, sol{2}(:,1,:), [], 2, 1);
axis on; axis equal; axis tight; colorbar;

function dg = dgnodes_of(mesh, porder, model, curved)
f = facenumbering(mesh.p, mesh.t, 1, mesh.boundaryexpr, []);
dg = createdgnodes(mesh.p, mesh.t, f, [], [], porder);
if curved
    if model == 1, w = dg(:,1,:)/0.5; else, w = (1 - dg(:,1,:))/0.5; end
    dg(:,1,:) = dg(:,1,:) + 0.05*sin(pi*dg(:,2,:)).*w;
end
end
