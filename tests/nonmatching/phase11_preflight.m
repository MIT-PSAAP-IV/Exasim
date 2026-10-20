function result = phase11_preflight(elemtype, ranksPerModel, interfaceType, porder, vertexPerm)
%PHASE11_PREFLIGHT Verify the existing conformal coupling path in 3D.
% elemtype = 0 uses tetrahedra; elemtype = 1 uses hexahedra.
% vertexPerm (optional) reorders the local vertices of every model-2 element,
% mesh{2}.t = mesh{2}.t(vertexPerm,:), to produce other relative interface
% face orientations (phase11_orientation_compare.m).

if nargin < 1, elemtype = 0; end
if nargin < 2, ranksPerModel = 1; end
if nargin < 3, interfaceType = 0; end
if nargin < 4, porder = 2; end
if nargin < 5, vertexPerm = []; end
assert(ismember(elemtype, [0 1]), 'elemtype must be 0 (tet) or 1 (hex)');
assert(ismember(ranksPerModel, [1 2]), 'preflight supports 1 or 2 ranks per model');

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
work = fullfile(tempdir, sprintf('exasim_phase11_preflight_e%d_%d_t%d', ...
    elemtype, ranksPerModel, interfaceType));
if exist(work, 'dir'), rmdir(work, 's'); end
mkdir(work);
old = pwd;
cleanup = onCleanup(@() cd(old));
cd(work);
addpath(here);
run(fullfile(repo, 'frontends', 'Matlab', 'exasim_setup.m'));

prefix = getenv('EXASIM_PHASE9_PREFIX');
if isempty(prefix), prefix = getenv('EXASIM_PHASE6_PREFIX'); end
if isempty(prefix), prefix = fullfile(fileparts(repo), 'exasim_install'); end
assert(isfile(fullfile(prefix, 'lib', 'cmake', 'Exasim', 'ExasimConfig.cmake')), ...
    'Phase 11 preflight requires an installed Exasim prefix');
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
    pde{model}.GMRESrestart = 200;
    pde{model}.linearsolveriter = 400;
    pde{model}.interfacetype = interfaceType;
    pde{model}.datapath = string(fullfile(work, 'data'));
    pde{model}.builddir = string(fullfile(work, 'build'));
    pde{model}.buildpath = pde{model}.builddir;
end

% Equal tangential grids make the x=0.5 interface exactly conformal. Retain a
% multi-element trace system in both rank layouts: the six tetrahedra of one
% logical cube form an ill-conditioned coupled trace fixture and cannot be
% partitioned meaningfully at 2+2.
if ranksPerModel == 1
    nxyz = [2,2,2];
else
    nxyz = [2,3,3];
end
[mesh{1}.p,mesh{1}.t] = cubemesh(nxyz(1),nxyz(2),nxyz(3),elemtype);
mesh{1}.p(1,:) = 0.5*mesh{1}.p(1,:);
[mesh{2}.p,mesh{2}.t] = cubemesh(nxyz(1),nxyz(2),nxyz(3),elemtype);
mesh{2}.p(1,:) = 0.5 + 0.5*mesh{2}.p(1,:);
if ~isempty(vertexPerm)
    mesh{2}.t = mesh{2}.t(vertexPerm,:);
end

mesh{1}.boundaryexpr = cube_boundaries(0, 0.5);
mesh{2}.boundaryexpr = cube_boundaries(0.5, 1);
mesh{1}.boundarycondition = [1;2;1;1;1;1];
mesh{2}.boundarycondition = [1;1;1;2;1;1];
mesh{1}.interfacecondition = [0;1;0;0;0;0];
mesh{2}.interfacecondition = [0;0;0;1;0;0];
pde{1}.interfacefluxmap = 1;
pde{2}.interfacefluxmap = 1;

referenceDgnodes = cell(2,1);
for model = 1:2
    f = facenumbering(mesh{model}.p, mesh{model}.t, elemtype, ...
        mesh{model}.boundaryexpr, mesh{model}.periodicexpr);
    referenceDgnodes{model} = createdgnodes(mesh{model}.p, mesh{model}.t, f, ...
        mesh{model}.curvedboundary, mesh{model}.curvedboundaryexpr, pde{model}.porder);
end

[sol,pde,mesh,master,dmd] = exasim(pde,mesh);
assert(master{1}.nd == 3 && master{2}.nd == 3, 'preflight did not construct 3D masters');
assert(master{1}.elemtype == elemtype && master{2}.elemtype == elemtype, ...
    'preflight element family differs from the requested family');

errors = zeros(2,1);
for model = 1:2
    assert(~isempty(sol{model}) && all(isfinite(sol{model}(:))), ...
        'incomplete conformal solution for model %d', model);
    if isfield(mesh{model}, 'dgnodes')
        dgnodes = mesh{model}.dgnodes;
    else
        dgnodes = referenceDgnodes{model};
    end
    x = dgnodes(:,1,:);
    y = dgnodes(:,2,:);
    z = dgnodes(:,3,:);
    exact = exp(x).*sin(pi*y).*sin(pi*z);
    difference = sol{model}(:,1,:)-exact;
    errors(model) = norm(difference(:))/max(norm(exact(:)), 1);
end

family = 'tet'; if elemtype == 1, family = 'hex'; end
fprintf('PASS phase11_preflight %s type %d %d+%d relative errors %.17e %.17e\n', ...
    family, interfaceType, ranksPerModel, ranksPerModel, errors(1), errors(2));
result = struct('family', family, 'ranks', ranksPerModel, 'errors', errors, ...
    'sol', {sol}, 'pde', {pde}, 'mesh', {mesh}, 'master', {master}, 'dmd', {dmd});
end

function expr = cube_boundaries(xmin, xmax)
expr = {@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-xmax)<1e-8, ...
    @(p) abs(p(2,:)-1)<1e-8, @(p) abs(p(1,:)-xmin)<1e-8, ...
    @(p) abs(p(3,:))<1e-8, @(p) abs(p(3,:)-1)<1e-8};
end
