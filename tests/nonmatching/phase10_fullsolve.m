function result = phase10_fullsolve(orientation, k1, k2, porder, N, composite, meshPair, curved, gap)
%PHASE10_FULLSOLVE Mixed triangle/quad coupled Poisson solve.
% orientation is 'tri-quad' or 'quad-tri'. meshPair is A (3:2), S (2:3),
% or R (nested 2:1). curved independently interpolates the same analytic wall.
if nargin < 1, orientation = 'tri-quad'; end
if nargin < 2, k1 = 1; end
if nargin < 3, k2 = k1; end
if nargin < 4, porder = 3; end
if nargin < 5, N = 1; end
if nargin < 6, composite = false; end
if nargin < 7, meshPair = 'A'; end
if nargin < 8, curved = false; end
if nargin < 9, gap = 0; end

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
example = fullfile(repo, 'examples', 'CoupledProblems', 'Poisson', 'mixedmeshes');
addpath(example); addpath(here);
work = fullfile(tempdir, sprintf('exasim_phase10_%s_%s_a1%d_c%d_g%+.4f_p%d_N%d_%d_%d', ...
    orientation, meshPair, composite, curved, gap, porder, N, k1, k2));
if exist(work, 'dir'), rmdir(work, 's'); end
mkdir(work); old = pwd; cleanup = onCleanup(@() cd(old)); cd(work);
run(fullfile(repo, 'frontends', 'Matlab', 'exasim_setup.m'));
prefix = getenv('EXASIM_PHASE9_PREFIX');
if isempty(prefix), prefix = getenv('EXASIM_PHASE6_PREFIX'); end
assert(~isempty(prefix), 'Set EXASIM_PHASE9_PREFIX to the tested installation');
setenv('EXASIM_PREFIX', prefix);

triangleOnModel1 = strcmp(orientation, 'tri-quad');
assert(triangleOnModel1 || strcmp(orientation, 'quad-tri'), 'unknown orientation');
pdeapp1; pdeapp2;
ranks = [k1 k2];
for m = 1:2
    pde{m}.mpiprocs = ranks(m);
    pde{m}.porder = porder;
    pde{m}.NLiter = 5;
    pde{m}.NLtol = 1e-13;
    pde{m}.linearsolvertol = 1e-13;
    pde{m}.matvectol = 1e-13;
    pde{m}.datapath = string(fullfile(work, 'data'));
    pde{m}.builddir = string(fullfile(work, 'build'));
    pde{m}.buildpath = pde{m}.builddir;
    if composite, pde{m}.phase9composite = true; end
end
if ~exist(fullfile(work, 'data'), 'dir'), mkdir(fullfile(work, 'data')); end

elem1 = double(~triangleOnModel1); elem2 = double(triangleOnModel1);
switch meshPair
    case 'A', cells1 = [2 4]*N; cells2 = [3 6]*N;
    case 'S', cells1 = [3 6]*N; cells2 = [2 4]*N;
    case 'R', cells1 = [2 4]*N; cells2 = [4 8]*N;
    otherwise, error('unknown mesh pair %s', meshPair);
end
[mesh{1}.p,mesh{1}.t] = squaremesh(cells1(1),cells1(2),1,elem1);
mesh{1}.p(1,:) = 0.5*mesh{1}.p(1,:);
[mesh{2}.p,mesh{2}.t] = squaremesh(cells2(1),cells2(2),1,elem2);
mesh{2}.p(1,:) = 0.5*mesh{2}.p(1,:) + 0.5 + gap;
mesh{2}.boundaryexpr = {@(q) abs(q(2,:))<1e-8, @(q) abs(q(1,:)-(1+gap))<1e-8, ...
    @(q) abs(q(2,:)-1)<1e-8, @(q) abs(q(1,:)-(0.5+gap))<1e-8};
for m = 1:2
    localElemtype = double(size(mesh{m}.t,1) == 4);
    f = facenumbering(mesh{m}.p, mesh{m}.t, localElemtype, mesh{m}.boundaryexpr, []);
    mesh{m}.dgnodes = createdgnodes(mesh{m}.p, mesh{m}.t, f, [], [], porder);
    if curved
        if m == 1, w = mesh{m}.dgnodes(:,1,:)/0.5;
        else, w = (1+gap-mesh{m}.dgnodes(:,1,:))/0.5; end
        mesh{m}.dgnodes(:,1,:) = mesh{m}.dgnodes(:,1,:) + ...
            0.05*sin(pi*mesh{m}.dgnodes(:,2,:)).*w;
    end
end
result = phase9_fullsolve('run', pde, mesh);
assert(result.status == 0, 'Phase 10 mixed solve failed');
assert(all(result.donor), 'Phase 10 did not run both donor directions');
assert(result.master{1}.elemtype ~= result.master{2}.elemtype, ...
    'Phase 10 masters do not use different element families');
fprintf('PASS phase10_fullsolve %s %s curved=%d gap=%+.3e p=%d N=%d %d+%d final residual %.3e\n', ...
    orientation, meshPair, curved, gap, porder, N, k1, k2, result.history.residuals(end));
end
