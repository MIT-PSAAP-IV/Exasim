% Four-domain non-matching Poisson example with hex-tet-hex-tet meshes.
% The exact solution is u = sin(pi*x) sin(pi*y) sin(pi*z).

clear porder N domainranks outdir result;

% User controls.
porder = 1;
N = 2;
domainranks = [1 1 1 1];
outdir = tempname;

% Generate, compile, solve, archive the run under /tmp, and report errors.
result = run_multiple_interactions(porder,N,domainranks,outdir);

% Plot nodal values on the z=0.5 plane.
figure(1); clf; hold on;
for m = 1:4
    x = reshape(result.mesh{m}.dgnodes(:,1,:),[],1);
    y = reshape(result.mesh{m}.dgnodes(:,2,:),[],1);
    z = reshape(result.mesh{m}.dgnodes(:,3,:),[],1);
    u = reshape(result.sol{m}(:,1,:),[],1);
    onPlane = abs(z-0.5)<1e-10;
    scatter(x(onPlane),y(onPlane),28,u(onPlane),'filled');
end
hold off; axis on; axis equal; axis tight; colorbar;
xlabel('x'); ylabel('y'); title('u(x,y,z=0.5)');
