% Generalized non-matching interaction between triangle and quadrilateral meshes.
% This is the interaction-graph counterpart of ../mixedmeshes.  Keeping the
% meshes, PDE models, polynomial order, and [2,3] rank split identical makes
% the two examples a direct regression pair.

clear pde mesh master dmd sol interactions;
here = fileparts(mfilename('fullpath'));
run(fullfile(here,'..','..','..','..','frontends','Matlab','exasim_setup.m'));

% User controls.
triangleOnModel1 = true;       % false reverses the triangle/quad orientation
curved = false;
domainranks = [2,3];
runDirectory = fullfile(tempdir,'exasim_multiple_interactions_mixedmeshes');

% Define the two domains and the generalized type-1 interaction.
pdeapp1;
pdeapp2;
interactions = interaction();

for m = 1:2
    mesh{m}.dgnodes = dgnodes_of(mesh{m},pde{m}.porder,m,curved);
end

% Generate, compile, and solve through the generalized interaction path.
[sol,pde,mesh,master,dmd] = exasim(pde,mesh,interactions);

% Report the manufactured-solution error.
err = 0;
for m = 1:2
    mesh{m}.porder = pde{m}.porder;
    mesh{m}.dgnodes = dgnodes_of(mesh{m},pde{m}.porder,m,curved);
    x = mesh{m}.dgnodes;
    e = sol{m}(:,1,:) - sin(pi*x(:,1,:)).*sin(pi*x(:,2,:));
    err = max(err,max(abs(e(:))));
end
fprintf('Generalized mixed-mesh maximum nodal error: %.3e\n',err);

figure(1); clf;
scaplot(mesh{1},sol{1}(:,1,:),[],2,1); hold on;
scaplot(mesh{2},sol{2}(:,1,:),[],2,1);
axis on; axis equal; axis tight; colorbar;

function dg = dgnodes_of(mesh,porder,model,curved)
elemtype = double(size(mesh.t,1)==4);
f = facenumbering(mesh.p,mesh.t,elemtype,mesh.boundaryexpr,[]);
dg = createdgnodes(mesh.p,mesh.t,f,[],[],porder);
if curved
    if model==1, w=dg(:,1,:)/0.5; else, w=(1-dg(:,1,:))/0.5; end
    dg(:,1,:) = dg(:,1,:) + 0.05*sin(pi*dg(:,2,:)).*w;
end
end
