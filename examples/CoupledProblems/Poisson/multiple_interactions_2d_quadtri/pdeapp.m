% Four-domain non-matching Poisson example with quad-tri-quad-tri meshes.
% The exact solution is u = sin(pi*x) sin(pi*y).

clear porder N domainranks outdir result;

% User controls.
porder = 1;
N = 2;
domainranks = [1 1 1 1];
outdir = tempname;

% Generate, compile, solve, archive the run under /tmp, and report errors.
result = run_multiple_interactions(porder,N,domainranks,outdir);

% Plot the four solutions together.
figure(1); clf;
for m = 1:4
    scaplot(result.mesh{m},result.sol{m}(:,1,:),[],2,1);
    hold on;
end
axis on; axis equal; axis tight; colorbar;
