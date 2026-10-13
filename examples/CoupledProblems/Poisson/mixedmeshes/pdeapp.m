% Mixed triangle/quadrilateral non-matching Poisson interface.
clear pde mesh master dmd sol;
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

triangleOnModel1 = true; % false reverses the triangle/quad orientation
curved = false;
pdeapp1;
pdeapp2;

for m = 1:2
    mesh{m}.dgnodes = dgnodes_of(mesh{m}, pde{m}.porder, m, curved);
end
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);

err = 0;
for m = 1:2
    mesh{m}.porder = pde{m}.porder;
    mesh{m}.dgnodes = dgnodes_of(mesh{m}, pde{m}.porder, m, curved);
    if numel(dmd{m}) == 1
        u = sol{m}; u(:,:,dmd{m}{1}.elempart(1:size(u,3))) = sol{m}; sol{m} = u;
    end
    x = mesh{m}.dgnodes;
    e = sol{m}(:,1,:) - sin(pi*x(:,1,:)).*sin(pi*x(:,2,:));
    err = max(err, max(abs(e(:))));
end
fprintf('Mixed triangle/quad maximum nodal error: %.3e\n', err);

figure(1); clf;
scaplot(mesh{1}, sol{1}(:,1,:), [], 2, 1); hold on;
scaplot(mesh{2}, sol{2}(:,1,:), [], 2, 1);
axis on; axis equal; axis tight; colorbar;

function dg = dgnodes_of(mesh, porder, model, curved)
elemtype = double(size(mesh.t,1) == 4);
f = facenumbering(mesh.p, mesh.t, elemtype, mesh.boundaryexpr, []);
dg = createdgnodes(mesh.p, mesh.t, f, [], [], porder);
if curved
    if model == 1, w = dg(:,1,:)/0.5; else, w = (1-dg(:,1,:))/0.5; end
    dg(:,1,:) = dg(:,1,:) + 0.05*sin(pi*dg(:,2,:)).*w;
end
end
