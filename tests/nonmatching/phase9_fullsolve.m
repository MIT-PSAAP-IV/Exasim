function result = phase9_fullsolve(caseName, k1, k2, suite)
%PHASE9_FULLSOLVE [nonmatching] Phase 9 coupled full-solve harness (generalizes Phase 6).
%
%   phase9_fullsolve('matching',  k1, k2)  type 0 vs type 1 on examples/CoupledProblems/Poisson/matching
%   phase9_fullsolve('nonlinear', k1, k2)  type 1 on the gapped (d ~= 0) nonlinear fixture
%   phase9_fullsolve('bypass')             gapped coordinates: type 0 must fail the backend
%                                          exact match, type 1 must set up both directions
%   phase9_fullsolve('partition', name, suite)  compare the stored 'matching' or 'nonlinear'
%                                          runs of one suite; the exact split set is required:
%                                          matching {1+1,1+3,2+2,3+1,4+4}, nonlinear
%                                          {1+1,2+2,3+1,4+4}
%
% k1/k2 are the MPI ranks of model 1/model 2 (unequal splits allowed). Every case uses
% the public field pde{m}.interfacetype (never pde.problem) and the production coupled
% compile call cmakecompile(pde{1}, k1 + k2), with no 1+1 workaround.
%
% Environment: EXASIM_PHASE9_PREFIX (or EXASIM_PHASE6_PREFIX) = Exasim install prefix.
% Results of 'matching'/'nonlinear' are stored per suite in
% tempdir/exasim_phase9_results/<suite> (suite: 4th argument, else $EXASIM_PHASE9_SUITE,
% else 'default') together with a configuration fingerprint; the 'partition' comparison
% reads only that suite, requires the exact split set and equal fingerprints.

if nargin < 1, caseName = 'matching'; end
if nargin < 2, k1 = 1; end
if nargin < 3, k2 = k1; end
here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
if strcmp(caseName, 'partition')
    if nargin < 3 || isempty(k2), k2 = suite_name([]); end
    result = compare_partitions(suite_dir(k2), k1);
    return;
end
if nargin < 4, suite = []; end
store = suite_dir(suite_name(suite));
if strcmp(caseName, 'run')
    % Shared runner for phase9_convergence.m: k1 = pde cell, k2 = mesh cell. Returns the
    % solutions in global element order plus master, mesh, dmd, log, the donor-search
    % flags of both directions and the parsed history.
    [sol, logtext, status, master, mesh, dmd] = run_coupled_case(k1, k2, false);
    result = struct('sol', {sol}, 'log', logtext, 'status', status, 'master', {master}, 'mesh', {mesh}, ...
        'dmd', {dmd}, 'donor', [has_donor_diagnostics(logtext, 1) has_donor_diagnostics(logtext, 2)]);
    if status == 0, result.history = parse_history(logtext); end
    return;
end

example = fullfile(repo, 'examples', 'CoupledProblems', 'Poisson', 'matching');
work = fullfile(tempdir, sprintf('exasim_phase9_%s_%d_%d', caseName, k1, k2));
if exist(work, 'dir'), rmdir(work, 's'); end
mkdir(work);
old = pwd;
cleanup = onCleanup(@() cd(old));
cd(work);
addpath(example);
addpath(here);
run(fullfile(repo, 'frontends', 'Matlab', 'exasim_setup.m'));
prefix = getenv('EXASIM_PHASE9_PREFIX');
if isempty(prefix), prefix = getenv('EXASIM_PHASE6_PREFIX'); end
assert(~isempty(prefix), 'Set EXASIM_PHASE9_PREFIX to the tested Exasim installation');
assert(isfile(fullfile(prefix, 'lib', 'cmake', 'Exasim', 'ExasimConfig.cmake')), ...
    'EXASIM_PHASE9_PREFIX is not an Exasim installation');
setenv('EXASIM_PREFIX', prefix);

switch caseName
    case 'matching'
        result = run_matching(work, k1, k2);
    case 'nonlinear'
        result = run_nonlinear(work, here, k1, k2);
    case 'bypass'
        result = run_bypass(work);
    otherwise
        error('Unknown case %s', caseName);
end
if ismember(caseName, {'matching', 'nonlinear'})
    result.fingerprint = config_fingerprint(prefix, here);
    save(fullfile(store, sprintf('%s_%d_%d.mat', caseName, k1, k2)), 'result');
end
end

function name = suite_name(suite)
if ~isempty(suite), name = char(suite); return; end
name = getenv('EXASIM_PHASE9_SUITE');
if isempty(name), name = 'default'; end
end

function d = suite_dir(name)
d = fullfile(tempdir, 'exasim_phase9_results', name);
if ~exist(d, 'dir'), mkdir(d); end
end

function fp = config_fingerprint(prefix, here)
% Content hash (MD5) of everything that determines a run, so runs of one suite can only be
% compared when they come from the same configuration (Revision 1, review N5):
% - the tested installation: every file under lib/ (static libraries, the text2code model
%   library, CMake package files; the Python site-packages are excluded), bin/, include/
%   and backend/ (sources compiled into the coupled application);
% - the MATLAB frontend used by the harness (frontends/Matlab, every .m file);
% - the harness and model files: tests/nonmatching/*.m and the coupled example directories.
% Each tree hash is the MD5 of "relative path:file MD5" over the sorted file list, so a
% renamed, added, removed or changed file changes the fingerprint; time stamps are not used.
repo = fileparts(fileparts(here));
parts = {char(prefix), ...
    tree_hash(fullfile(prefix, 'lib'), '*', 'python'), ...
    tree_hash(fullfile(prefix, 'bin'), '*', ''), ...
    tree_hash(fullfile(prefix, 'include'), '*', ''), ...
    tree_hash(fullfile(prefix, 'backend'), '*', ''), ...
    tree_hash(fullfile(repo, 'frontends', 'Matlab'), '*.m', ''), ...
    tree_hash(here, '*.m', ''), ...
    tree_hash(fullfile(repo, 'examples', 'CoupledProblems', 'Poisson', 'matching'), '*.m', ''), ...
    tree_hash(fullfile(repo, 'examples', 'CoupledProblems', 'Poisson', 'nonmatching'), '*.m', '')};
fp = strjoin(parts, '|');
end

function h = tree_hash(dirname, pattern, exclude)
dirname = char(dirname);
if ~isfolder(dirname), h = 'absent'; return; end
files = dir(fullfile(dirname, '**', pattern));
files = files(~[files.isdir]);
names = cellfun(@(f, d) fullfile(d, f), {files.name}, {files.folder}, 'UniformOutput', false);
if ~isempty(exclude)
    names = names(~contains(names, [filesep exclude]));
end
names = sort(names);
entries = cellfun(@(f) [strrep(f, dirname, '') ':' file_md5(f)], names, 'UniformOutput', false);
h = file_md5_text(strjoin(entries, ';'));
end

% ---------------------------------------------------------------------------
function result = run_matching(work, k1, k2)
logs = cell(2, 1); sols = cell(2, 1); hist = cell(2, 1);
for interfaceType = 0:1
    % Both interface types use identical generated kernels: one build, reused through the
    % kernel fingerprint in run_coupled_case (only the input directories differ).
    [pde, mesh] = matching_problem(interfaceType, k1, k2, fullfile(work, sprintf('type%d', interfaceType)), ...
        fullfile(work, 'build'));
    [sols{interfaceType + 1}, logs{interfaceType + 1}, status] = run_coupled_case(pde, mesh);
    assert(status == 0, 'type-%d solver exited with status %d', interfaceType, status);
    hist{interfaceType + 1} = parse_history(logs{interfaceType + 1});
end
assert(~has_donor_diagnostics(logs{1}, 1) && ~has_donor_diagnostics(logs{1}, 2), ...
    'type 0 executed the type-1 donor search');
assert(has_donor_diagnostics(logs{2}, 1) && has_donor_diagnostics(logs{2}, 2), ...
    'type 1 did not execute both donor-search directions');

[absErr, relErr] = solution_difference(sols{1}, sols{2});
fprintf('PHASE9 matching %d+%d: abs %.17e rel %.17e\n', k1, k2, absErr, relErr);
print_history(sprintf('PHASE9 matching %d+%d type0', k1, k2), hist{1});
print_history(sprintf('PHASE9 matching %d+%d type1', k1, k2), hist{2});

assert(isequal(hist{1}.iterations, hist{2}.iterations), 'Newton iteration sequences differ');
scale = max(abs(hist{1}.residuals), abs(hist{2}.residuals));
assert(all(abs(hist{1}.residuals - hist{2}.residuals) <= 1e-12 + 1e-6*scale), ...
    'type-0 and type-1 Newton residual histories differ');
assert(isequal(hist{1}.gmres, hist{2}.gmres), 'GMRES iteration counts differ');
assert(relErr <= 1e-12, 'matching type-0/type-1 solutions differ beyond 1e-12 relative');
for m = 1:2
    assert(all(isfinite(sols{2}{m}(:))) && ~isempty(sols{2}{m}), 'incomplete type-1 output, model %d', m);
end
fprintf('PASS phase9_fullsolve matching %d+%d\n', k1, k2);
result = struct('case', 'matching', 'k1', k1, 'k2', k2, 'sol', {sols{2}}, 'sol0', {sols{1}}, ...
    'history0', hist{1}, 'history1', hist{2}, 'relErr', relErr);
end

function [pde, mesh] = matching_problem(interfaceType, k1, k2, casepath, build)
pdeapp1;
pdeapp2;
pde{1}.modelnumber = 1;
pde{2}.modelnumber = 2;
mesh{1}.interfacecondition = [0; 1; 0; 0];
mesh{2}.interfacecondition = [0; 0; 0; 1];
pde{1}.interfacefluxmap = 1;
pde{2}.interfacefluxmap = 1;
ranks = [k1 k2];
for model = 1:2
    pde{model}.mpiprocs = ranks(model);
    pde{model}.porder = 2;
    pde{model}.NLiter = 4;
    pde{model}.NLtol = 1e-11;
    pde{model}.linearsolvertol = 1e-6;
    pde{model}.matvectol = 1e-6;
    pde{model}.GMRESrestart = 200;
    pde{model}.linearsolveriter = 400;
    pde{model}.interfacetype = interfaceType;
    pde = set_paths(pde, model, casepath, build);
end
end

% ---------------------------------------------------------------------------
function result = run_nonlinear(work, here, k1, k2)
[pde, mesh, minimumGap] = gapped_problem(1, k1, k2, fullfile(work, 'nonmatching'), true, here);
[sol, logtext, status] = run_coupled_case(pde, mesh);
assert(status == 0, 'nonlinear type-1 solver exited with status %d', status);
assert(has_donor_diagnostics(logtext, 1) && has_donor_diagnostics(logtext, 2), ...
    'nonlinear run did not execute both donor-search directions');
h = parse_history(logtext);
gaps = token_numbers(regexp(logtext, 'max distance =\s*([-+0-9.eE]+),\s*points off donor face', 'tokens'));
ratios = h.residuals(2:end) ./ (h.residuals(1:end-1).^2);
preFloor = h.residuals(1:end-1) > 1e-9;
print_history(sprintf('PHASE9 nonlinear %d+%d', k1, k2), h);
fprintf('PHASE9 nonlinear %d+%d: mesh gap %.17e, donor max distances', k1, k2, minimumGap);
fprintf(' %.17e', gaps); fprintf('\n');
fprintf('PHASE9 nonlinear %d+%d quadratic ratios:', k1, k2); fprintf(' %.8e', ratios); fprintf('\n');
assert(minimumGap > 9e-3, 'nonlinear fixture lost its physical gap');
assert(numel(gaps) >= 2 && all(abs(gaps - minimumGap) < 1e-10), 'donor distance does not equal the mesh gap');
assert(h.residuals(end) < 1e-10, 'nonlinear solve did not reach residual < 1e-10');
assert(nnz(ratios(preFloor) < 0.5) >= 2, 'fewer than two pre-floor quadratic ratios below 0.5');
fprintf('PASS phase9_fullsolve nonlinear %d+%d\n', k1, k2);
result = struct('case', 'nonlinear', 'k1', k1, 'k2', k2, 'sol', {sol}, 'history1', h, ...
    'ratios', ratios, 'gap', minimumGap);
end

function [pde, mesh, minimumGap] = gapped_problem(interfaceType, k1, k2, casepath, nonlinear, testdir)
pdeapp1;
pdeapp2;
if nonlinear
    addpath(testdir);
    pde{1}.modelfile = "pdemodel_phase6_1";
    pde{2}.modelfile = "pdemodel_phase6_2";
end
pde{1}.modelnumber = 1;
pde{2}.modelnumber = 2;
mesh{1}.interfacecondition = [0; 1; 0; 0];
pde{1}.interfacefluxmap = 1;
% Different transverse partition and a 0.01 gap exercise d ~= 0 (Phase 6 fixture).
[mesh{2}.p, mesh{2}.t] = squaremesh(7, 11, 1, 1);
mesh{2}.p(1,:) = 0.49*mesh{2}.p(1,:) + 0.51;
mesh{2}.boundaryexpr = {@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-1)<1e-8, ...
    @(p) abs(p(2,:)-1)<1e-8, @(p) abs(p(1,:)-0.51)<1e-8};
mesh{2}.boundarycondition = [1; 1; 1; 2];
mesh{2}.interfacecondition = [0; 0; 0; 1];
pde{2}.interfacefluxmap = 1;
minimumGap = min(mesh{2}.p(1,:)) - max(mesh{1}.p(1,:));
ranks = [k1 k2];
for model = 1:2
    pde{model}.mpiprocs = ranks(model);
    pde{model}.porder = 2;
    pde{model}.NLiter = 10;
    pde{model}.NLtol = 1e-12;
    pde{model}.linearsolvertol = 1e-11;
    pde{model}.matvectol = 1e-11;
    pde{model}.GMRESrestart = 250;
    pde{model}.linearsolveriter = 500;
    pde{model}.interfacetype = interfaceType;
    pde = set_paths(pde, model, casepath);
end
end

% ---------------------------------------------------------------------------
function result = run_bypass(work)
% Gapped coordinates cannot be matched exactly: the backend's conformal matcher
% (build_runtime_interface_partition) must stop type 0 on every rank, while type 1
% skips only that matcher and sets up both donor-search directions.
here = fileparts(mfilename('fullpath'));
[pde, mesh] = gapped_problem(0, 1, 1, fullfile(work, 'type0'), false, here);
[~, log0, status0] = run_coupled_case(pde, mesh, false);
[pde, mesh] = gapped_problem(1, 1, 1, fullfile(work, 'type1'), false, here);
[~, log1, status1] = run_coupled_case(pde, mesh);
type0Rejected = status0 ~= 0 && contains(log0, 'no coupled interface match');
fprintf('PHASE9 bypass: type0 status %d (match error %d), type1 status %d (diagnostics %d/%d)\n', ...
    status0, type0Rejected, status1, has_donor_diagnostics(log1, 1), has_donor_diagnostics(log1, 2));
assert(type0Rejected, 'type 0 did not fail the backend exact interface match on gapped coordinates');
assert(status1 == 0 && has_donor_diagnostics(log1, 1) && has_donor_diagnostics(log1, 2), ...
    'type 1 did not set up both donor-search directions on gapped coordinates');
fprintf('PASS phase9_fullsolve bypass\n');
result = struct('case', 'bypass', 'status0', status0, 'status1', status1);
end

% ---------------------------------------------------------------------------
function result = compare_partitions(store, name)
switch name
    case 'matching',  expected = {'1+1', '1+3', '2+2', '3+1', '4+4'};
    case 'nonlinear', expected = {'1+1', '2+2', '3+1', '4+4'};
    otherwise, error('unknown partition set %s', name);
end
files = dir(fullfile(store, [name '_*.mat']));
runs = cell(numel(files), 1);
for i = 1:numel(files)
    data = load(fullfile(files(i).folder, files(i).name));
    runs{i} = data.result;
end
labels = cellfun(@(r) sprintf('%d+%d', r.k1, r.k2), runs, 'UniformOutput', false);
assert(isequal(sort(labels(:)).', sort(expected)), 'suite %s holds splits {%s}, expected {%s}', ...
    store, strjoin(labels, ','), strjoin(expected, ','));
fps = cellfun(@(r) r.fingerprint, runs, 'UniformOutput', false);
assert(all(strcmp(fps, fps{1})), 'suite %s mixes runs of different configurations', store);
reference = find(strcmp(labels, '1+1'), 1);
if isempty(reference), reference = 1; end
tol = 1e-12;
if strcmp(name, 'nonlinear'), tol = 1e-10; end
worst = 0;
for i = 1:numel(runs)
    % run_coupled_case stores every model in the partition-independent global
    % element order (dmd.elempart, also for single-rank models); equal sizes are
    % asserted before comparing.
    for m = 1:2
        assert(isequal(size(runs{i}.sol{m}), size(runs{reference}.sol{m})), ...
            'split %s model %d has a different global size', labels{i}, m);
    end
    [~, relErr] = solution_difference(runs{reference}.sol, runs{i}.sol);
    fprintf('PHASE9 partition %s: %s vs %s rel %.3e\n', name, labels{i}, labels{reference}, relErr);
    worst = max(worst, relErr);
end
assert(worst <= tol, 'partition dependence %.3e exceeds %.1e', worst, tol);
fprintf('PASS phase9_fullsolve partition %s (max rel %.3e over %d splits)\n', name, worst, numel(runs));
result = struct('case', 'partition', 'worst', worst, 'labels', {labels});
end

% ---------------------------------------------------------------------------
function pde = set_paths(pde, model, casepath, build)
pde{model}.datapath = string(casepath);
if nargin < 4, build = string(casepath) + "/.exasim"; end
pde{model}.builddir = string(build);
pde{model}.buildpath = pde{model}.builddir;
if model == 1 && ~exist(casepath, 'dir'), mkdir(casepath); end
end

function [sol, logtext, status, master, mesh, dmd] = run_coupled_case(pde, mesh, mustSucceed)
% The legacy two-model branch of exasim.m with a checked solver launch: the
% compile call is exactly the production one, cmakecompile(pde{1}, total ranks).
if nargin < 3, mustSucceed = true; end
nmodels = 2;
master = cell(nmodels, 1);
dmd = cell(nmodels, 1);
sol = cell(nmodels, 1);
total = pde{1}.mpiprocs + pde{2}.mpiprocs;
for model = 1:nmodels
    pde{model}.combinedmodel = false;
    [pde{model}, mesh{model}, master{model}, dmd{model}] = preprocessing(pde{model}, mesh{model});
end
% Generate the kernels, then compile only if the generated kernel set, the coupled
% variant or the install contents (lib, include, backend; MD5) differ from the build
% already in pde{1}.builddir (the kernels
% do not depend on the mesh, the rank split, the polynomial order or interfacetype).
exe = string(pde{1}.builddir) + "/build/exasimapp";
for model = 1:nmodels
    kkgencode(pde{model});
end
kernels = string(pde{1}.builddir) + "/kernels";
kkgencodeall(nmodels, kernels);
prefix = getenv('EXASIM_PREFIX');
fingerprint = [kernel_fingerprint(char(kernels)) '|mpi=' num2str(total > 1) '|' prefix '|' ...
    tree_hash(fullfile(prefix, 'lib'), '*', 'python') '|' tree_hash(fullfile(prefix, 'include'), '*', '') ...
    '|' tree_hash(fullfile(prefix, 'backend'), '*', '')];
stamp = char(string(pde{1}.builddir) + "/phase9_build_fingerprint.txt");
if isfile(exe) && isfile(stamp) && strcmp(strtrim(fileread(stamp)), fingerprint)
    fprintf('PHASE9 build reused (kernel fingerprint match): %s\n', exe);
else
    cmakecompile(pde{1}, total);
    fid = fopen(stamp, 'w'); fprintf(fid, '%s\n', fingerprint); fclose(fid);
end
rendered = fileread(char(string(pde{1}.builddir) + "/CMakeLists.txt"));
assert(contains(rendered, 'cpumpi'), 'coupled %d-rank build did not select the MPI variant', total);
[status, logtext] = run_solver(pde{1}, 100 + pde{1}.mpiprocs, total, exe);
fprintf('%s', logtext);
if status ~= 0
    assert(~mustSucceed, 'solver failed with exit status %d', status);
    return;
end
for model = 1:nmodels
    sol{model} = global_order(fetchsolution(pde{model}, master{model}, dmd{model}, ...
        pde{model}.datapath + "/dataout" + num2str(model)), dmd{model});
end
end

function fp = kernel_fingerprint(dirname)
files = dir(fullfile(dirname, '**', '*'));
files = files(~[files.isdir]);
names = sort(cellfun(@(f, d) fullfile(d, f), {files.name}, {files.folder}, 'UniformOutput', false));
parts = cellfun(@(f) [strrep(f, dirname, '') ':' file_md5(f)], names, 'UniformOutput', false);
fp = file_md5_text(strjoin(parts, ';'));
end

function h = file_md5(filename)
fid = fopen(filename, 'r'); bytes = fread(fid, inf, '*uint8'); fclose(fid);
h = md5_bytes(bytes);
end

function h = file_md5_text(text)
h = md5_bytes(uint8(char(text)));
end

function h = md5_bytes(bytes)
md = java.security.MessageDigest.getInstance('MD5');
if ~isempty(bytes), md.update(bytes(:)); end
h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2).', 1, []));
end

function s = global_order(s, d)
% getsolutions places the elements of every rank at dmd.elempart when a model has
% several ranks, but returns a single-rank model in that rank's local element order.
% Apply elempart in that case too, so every split is in the partition-independent
% global element order of the model mesh.
if iscell(d) && numel(d) == 1 && isfield(d{1}, 'elempart') && ~isempty(d{1}.elempart)
    ne = size(s, 3);
    ep = d{1}.elempart(:);
    assert(numel(ep) >= ne && isequal(sort(ep(1:ne)).', 1:ne), 'single-rank elempart is not a permutation');
    g = zeros(size(s));
    g(:, :, ep(1:ne), :) = s;
    s = g;
end
end

function [status, output] = run_solver(pde, numpde, mpiprocs, exe)
if nargin < 4, exe = string(pde.builddir) + "/build/exasimapp"; end
exe = string(exe);
assert(isfile(exe), 'Solver executable is missing: %s', exe);
mpirun = string(pde.mpirun);
mpitxt = char(string(pde.builddir) + "/build/mpiexec.txt");
if isfile(mpitxt)
    discovered = strip(string(fileread(mpitxt)));
    if strlength(discovered) > 0, mpirun = discovered; end
end
runstr = mpirun + " --oversubscribe -np " + string(mpiprocs) + " " + shell_quote(exe) + ...
    " " + string(numpde) + " " + ...
    shell_quote(string(pde.datapath) + "/datain1/") + " " + ...
    shell_quote(string(pde.datapath) + "/dataout1/out") + " " + ...
    shell_quote(string(pde.datapath) + "/datain2/") + " " + ...
    shell_quote(string(pde.datapath) + "/dataout2/out");
[status, output] = system(char(runstr + " 2>&1"));
end

function quoted = shell_quote(value)
value = string(value);
assert(~contains(value, "'"), 'Phase 9 test paths may not contain a single quote');
quoted = "'" + value + "'";
end

function yes = has_donor_diagnostics(logtext, direction)
yes = contains(logtext, sprintf('Non-matching donor search, Gamma^%d', direction));
end

function h = parse_history(logtext)
% Initial residual, then one accepted step per Newton iteration: the solver prints a
% line per backtracking trial and a final line with the accepted alpha and norm.
initial = regexp(logtext, 'Newton Iteration:\s*0,\s*Residual Norm:\s*([-+0-9.eE]+)', 'tokens', 'once');
assert(~isempty(initial), 'initial Newton residual not found');
lines = regexp(logtext, ['Newton Iteration:\s*([0-9]+),\s*Alpha:\s*([-+0-9.eE]+),\s*Original Norm:' ...
    '\s*([-+0-9.eE]+),\s*Updated Norm:\s*([-+0-9.eE]+)'], 'tokens');
it = cellfun(@(t) str2double(t{1}), lines);
[uniqueIt, last] = unique(it, 'last');
h.iterations = [0 uniqueIt(:).'];
h.residuals = [str2double(initial{1}) cellfun(@(t) str2double(t{4}), lines(last))];
h.alphas = cellfun(@(t) str2double(t{2}), lines(last));
h.backtracks = numel(lines) - numel(last);
gm = regexp(logtext, 'GMRES\(([0-9]+)\) converges[^\r\n]*?within\s*([0-9]+) iterations', 'tokens');
h.gmres = cellfun(@(t) str2double(t{2}), gm);
gmLines = regexp(logtext, 'GMRES\([0-9]+\) converges[^\r\n]*', 'match');
h.gmresLines = gmLines;
assert(isequal(h.iterations, 0:numel(h.residuals)-1), 'Newton iteration sequence is incomplete');
end

function print_history(label, h)
fprintf('%s Newton residuals:', label); fprintf(' %.6e', h.residuals); fprintf('\n');
fprintf('%s accepted steps:', label); fprintf(' %g', h.alphas); fprintf(' (backtracks %d)\n', h.backtracks);
fprintf('%s GMRES iterations:', label); fprintf(' %d', h.gmres); fprintf('\n');
for i = 1:numel(h.gmresLines), fprintf('%s %s\n', label, strtrim(h.gmresLines{i})); end
end

function [absErr, relErr] = solution_difference(a, b)
absErr = 0; ref = 0;
for m = 1:numel(a)
    absErr = max(absErr, norm(b{m}(:) - a{m}(:), inf));
    ref = max(ref, norm(a{m}(:), inf));
end
relErr = absErr / max(ref, realmin);
end

function values = token_numbers(tokens)
values = cellfun(@(t) str2double(t{1}), tokens);
end
