function result = phase11_fullsolve(family, configuration, porder, N, ranksPerModel, referenceMaxu)
%PHASE11_FULLSOLVE Focused 3D non-matching Poisson verification case.
%
% Fixed criteria (declared before the first solve): M1 last-level rate >=
% p+1-0.2, partition relative difference <=5e-12, linear tolerance <=1e-12.
% M2 is diagnostic under S1(a). M3 cases are diagnostic (rates reported) with
% analytic envelopes declared here before their first solve:
%   M3  gapped:      model 1 on [0,0.49], model 2 on [0.51,1] (gap 0.02);
%   M3o overlapping: model 1 on [0,0.51], model 2 on [0.49,1] (overlap 0.02);
%   M3c curved:      interface x = 0.5 + 0.05 sin(pi y) sin(pi z), each side
%                    interpolating the curve with its own DG nodes on N x 2N x 2N
%                    (model 1) and N x N x N (model 2) cells.
% Envelopes: max|u_h| <= max|u_h(M1)| + 0.05 at the same family, p and N,
% passed as referenceMaxu (maintainer decision 2026-10-15; the original
% absolute bound 1.05 exp(1) is violated by the matching M1 runs themselves at
% coarse levels through nodal overshoot); the maximum donor distance is
% <= 0.02 + 1e-12 for M3 and M3o and <= 0.05 (pi h)^2 with h = 1/N for M3c
% (two-sided interpolation of the curved interface).

if nargin < 1, family = 'tet'; end
if nargin < 2, configuration = 'M1'; end
if nargin < 3, porder = 2; end
if nargin < 4, N = 2; end
if nargin < 5, ranksPerModel = 1; end
if nargin < 6, referenceMaxu = []; end
assert(ismember(family, {'tet','hex'}), 'family must be tet or hex');
assert(ismember(configuration, {'M0','M1','M2','M3','M3o','M3c'}), 'configuration must be M0, M1, M2, M3, M3o or M3c');
assert(ismember(ranksPerModel, [1 2 4]), 'ranksPerModel must be 1, 2 or 4');
elemtype = strcmp(family, 'hex');

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
work = fullfile(tempdir, sprintf('exasim_phase11_%s_%s_p%d_N%d_r%d', ...
    family, configuration, porder, N, ranksPerModel));
if exist(work, 'dir'), rmdir(work, 's'); end
mkdir(work); old = pwd; cleanup = onCleanup(@() cd(old)); cd(work);
addpath(here); run(fullfile(repo, 'frontends', 'Matlab', 'exasim_setup.m'));
prefix = getenv('EXASIM_PHASE9_PREFIX');
if isempty(prefix), prefix = fullfile(fileparts(repo), 'Exasim-build', 'install'); end
assert(isfile(fullfile(prefix, 'lib', 'cmake', 'Exasim', 'ExasimConfig.cmake')), ...
    'Phase 11 requires an installed Exasim prefix');
setenv('EXASIM_PREFIX', prefix);

pde = cell(2,1); mesh = cell(2,1);
for model = 1:2
    [pde{model},mesh{model}] = initializeexasim();
    pde{model}.model = "ModelD";
    pde{model}.modelfile = "pdemodel_phase11_" + string(model);
    pde{model}.modelnumber = model;
    pde{model}.Cxxpreprocessing = 1;
    pde{model}.platform = "cpu";
    pde{model}.mpiprocs = ranksPerModel;
    pde{model}.hybrid = 1;
    pde{model}.porder = porder;
    pde{model}.pgauss = 2*porder;
    pde{model}.physicsparam = 1;
    pde{model}.tau = 1;
    pde{model}.NLiter = 4;
    pde{model}.NLtol = 1e-12;
    pde{model}.linearsolvertol = 1e-12;
    pde{model}.matvectol = 1e-12;
    pde{model}.GMRESrestart = 250;
    pde{model}.linearsolveriter = 600;
    pde{model}.interfacetype = double(~strcmp(configuration, 'M0'));
    pde{model}.datapath = string(fullfile(work, 'data'));
    pde{model}.builddir = string(fullfile(work, 'build'));
    pde{model}.buildpath = pde{model}.builddir;
end

switch configuration
    case 'M0', ny = [N N]; nz = [N N]; bounds = [0.5 0.5];
    case 'M1', ny = [2*N N]; nz = [2*N N]; bounds = [0.5 0.5];
    case 'M2', ny = [N 2*N]; nz = [N 2*N]; bounds = [0.5 0.5];
    case 'M3', ny = [N N]; nz = [N N]; bounds = [0.49 0.51];
    case 'M3o', ny = [N N]; nz = [N N]; bounds = [0.51 0.49];
    case 'M3c', ny = [2*N N]; nz = [2*N N]; bounds = [0.5 0.5];
end
curved = strcmp(configuration, 'M3c');
[mesh{1}.p,mesh{1}.t] = cubemesh(N,ny(1),nz(1),elemtype);
mesh{1}.p(1,:) = bounds(1)*mesh{1}.p(1,:);
[mesh{2}.p,mesh{2}.t] = cubemesh(N,ny(2),nz(2),elemtype);
mesh{2}.p(1,:) = bounds(2)+(1-bounds(2))*mesh{2}.p(1,:);
mesh{1}.boundaryexpr = cube_boundaries(0,bounds(1));
mesh{2}.boundaryexpr = cube_boundaries(bounds(2),1);
mesh{1}.boundarycondition = [1;2;1;1;1;1];
mesh{2}.boundarycondition = [1;1;1;2;1;1];
mesh{1}.interfacecondition = [0;1;0;0;0;0];
mesh{2}.interfacecondition = [0;0;0;1;0;0];
pde{1}.interfacefluxmap = 1; pde{2}.interfacefluxmap = 1;

% Reference DG nodes: the multi-rank frontend path does not return
% mesh.dgnodes, so build them from the same vertex mesh (as phase11_preflight).
referenceDgnodes = cell(2,1);
for model = 1:2
    f = facenumbering(mesh{model}.p, mesh{model}.t, elemtype, ...
        mesh{model}.boundaryexpr, mesh{model}.periodicexpr);
    referenceDgnodes{model} = createdgnodes(mesh{model}.p, mesh{model}.t, f, ...
        mesh{model}.curvedboundary, mesh{model}.curvedboundaryexpr, porder);
    if curved
        % Bend the interface; the map fixes the outer boundaries.
        dg = referenceDgnodes{model};
        if model == 1, w = dg(:,1,:)/0.5; else, w = (1 - dg(:,1,:))/0.5; end
        dg(:,1,:) = dg(:,1,:) + 0.05*sin(pi*dg(:,2,:)).*sin(pi*dg(:,3,:)).*w;
        referenceDgnodes{model} = dg;
        mesh{model}.dgnodes = dg;   % preprocessing uses mesh.dgnodes when present
    end
end

logtext = evalc('[sol,pde,mesh,master,dmd] = exasim(pde,mesh);');
assert(contains(logtext, 'GMRES') && all(cellfun(@(s) ~isempty(s), sol)), ...
    'Phase 11 coupled solve did not complete');
if pde{1}.interfacetype == 1
    assert(contains(logtext, 'Non-matching donor search, Gamma^1') && ...
           contains(logtext, 'Non-matching donor search, Gamma^2'), ...
           'type-1 donor search did not execute in both directions');
end
% Solutions and DG nodes in global element order: a one-rank model returns its
% solution in partition order (as the 2D coupled examples handle it).
for model = 1:2
    if numel(dmd{model}) == 1
        u = sol{model};
        u(:,:,dmd{model}{1}.elempart(1:size(u,3))) = sol{model};
        sol{model} = u;
    end
    mesh{model}.dgnodes = referenceDgnodes{model};
end
[e1,n1] = l2error3d(sol{1},mesh{1}.dgnodes,master{1});
[e2,n2] = l2error3d(sol{2},mesh{2}.dgnodes,master{2});
maxu = max(max(abs(sol{1}(:,1,:)),[],'all'), max(abs(sol{2}(:,1,:)),[],'all'));
% Converged-solve requirement (predeclared): final Newton residual <= 1e-11.
nwt = regexp(logtext, 'Updated Norm:\s*([-+0-9.eE]+)', 'tokens');
finalNewton = NaN;
if ~isempty(nwt), finalNewton = str2double(nwt{end}{1}); end
assert(finalNewton <= 1e-11, '%s %s p=%d N=%d: final Newton residual %.3e exceeds 1e-11', ...
    family, configuration, porder, N, finalNewton);
dist = regexp(logtext, 'max distance = ([-+0-9.eE]+)', 'tokens');
maxdist = 0;
if ~isempty(dist), maxdist = max(cellfun(@(c) str2double(c{1}), dist)); end
if startsWith(configuration,'M3')
    assert(~isempty(referenceMaxu), 'M3 cases need referenceMaxu = max|u_h(M1)| at the same family, p and N');
    assert(maxu <= referenceMaxu + 0.05, '%s max|u| %.6f exceeds max|u(M1)| + 0.05 = %.6f', ...
        configuration, maxu, referenceMaxu + 0.05);
    if curved, bound = 0.05*(pi/N)^2; else, bound = 0.02 + 1e-12; end
    assert(maxdist <= bound, '%s donor distance %.3e exceeds the declared bound %.3e', configuration, maxdist, bound);
end
result = struct('family',family,'configuration',configuration,'p',porder,'N',N, ...
    'ranks',ranksPerModel,'e1',e1,'e2',e2,'e',hypot(e1,e2),'ndof',n1+n2, ...
    'maxu',maxu,'gap',bounds(2)-bounds(1),'maxdist',maxdist,'finalNewton',finalNewton,'sol',{sol},'mesh',{mesh}, ...
    'master',{master},'dmd',{dmd},'log',logtext);
fprintf('PHASE11 %s %s p=%d N=%d ranks=%d+%d dofs=%d e1=%.9e e2=%.9e e=%.9e gap=%.3e maxdist=%.3e maxu=%.9e\n', ...
    family,configuration,porder,N,ranksPerModel,ranksPerModel,result.ndof,e1,e2,result.e,result.gap,maxdist,maxu);
end

function [err,ndof] = l2error3d(u,dgnodes,master)
phi = master.shapegt(:,:,1);
dr = master.shapegt(:,:,2); ds = master.shapegt(:,:,3); dt = master.shapegt(:,:,4);
w = master.gwe(:); [npe,~,ne] = size(dgnodes); ndof = npe*ne; err = 0;
for e = 1:ne
    X = dgnodes(:,:,e); xg = phi*X;
    xr=dr*X; xs=ds*X; xt=dt*X;
    jac = xr(:,1).*(xs(:,2).*xt(:,3)-xs(:,3).*xt(:,2)) ...
        - xr(:,2).*(xs(:,1).*xt(:,3)-xs(:,3).*xt(:,1)) ...
        + xr(:,3).*(xs(:,1).*xt(:,2)-xs(:,2).*xt(:,1));
    uh = phi*u(:,1,e);
    ue = exp(xg(:,1)).*sin(pi*xg(:,2)).*sin(pi*xg(:,3));
    err = err + sum(w.*abs(jac).*(uh-ue).^2);
end
err = sqrt(err);
end

function expr = cube_boundaries(xmin,xmax)
expr = {@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-xmax)<1e-8, ...
    @(p) abs(p(2,:)-1)<1e-8, @(p) abs(p(1,:)-xmin)<1e-8, ...
    @(p) abs(p(3,:))<1e-8, @(p) abs(p(3,:)-1)<1e-8};
end
