% Mixed triangle/quadrilateral non-matching Poisson interface with a 0.02 gap.
clear pde mesh master dmd sol;
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

triangleOnModel1 = true; % false reverses the triangle/quad orientation
curved = false;
gap = 0.02;
pdeapp1;
pdeapp2;

actualGap = min(mesh{2}.p(1,:)) - max(mesh{1}.p(1,:));
assert(abs(actualGap-gap) < 1e-12, 'Expected interface gap %.16g, found %.16g.', gap, actualGap);
fprintf('Mixed triangle/quad interface gap: %.6f\n', actualGap);

for m = 1:2
    mesh{m}.dgnodes = dgnodes_of(mesh{m}, pde{m}.porder, m, curved, gap);
end
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);

err = 0;
for m = 1:2
    mesh{m}.porder = pde{m}.porder;
    mesh{m}.dgnodes = dgnodes_of(mesh{m}, pde{m}.porder, m, curved, gap);
    if numel(dmd{m}) == 1
        u = sol{m}; u(:,:,dmd{m}{1}.elempart(1:size(u,3))) = sol{m}; sol{m} = u;
    end
    x = mesh{m}.dgnodes;
    e = sol{m}(:,1,:) - sin(pi*x(:,1,:)).*sin(pi*x(:,2,:));
    err = max(err, max(abs(e(:))));
end
fprintf('Gapped mixed triangle/quad maximum nodal error: %.3e\n', err);

figure(1); clf;
scaplot(mesh{1}, sol{1}(:,1,:), [], 2, 1); hold on;
scaplot(mesh{2}, sol{2}(:,1,:), [], 2, 1);
axis on; axis equal; axis tight; colorbar;

function dg = dgnodes_of(mesh, porder, model, curved, gap)
elemtype = double(size(mesh.t,1) == 4);
f = facenumbering(mesh.p, mesh.t, elemtype, mesh.boundaryexpr, []);
dg = createdgnodes(mesh.p, mesh.t, f, [], [], porder);
if curved
    width = 0.5-gap/2;
    if model == 1, w = dg(:,1,:)/width; else, w = (1-dg(:,1,:))/width; end
    dg(:,1,:) = dg(:,1,:) + 0.05*sin(pi*dg(:,2,:)).*w;
end
end
