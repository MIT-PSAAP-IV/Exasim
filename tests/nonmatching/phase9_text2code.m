function result = phase9_text2code(k, exampleName, orientation)
%PHASE9_TEXT2CODE [nonmatching] Phase 9 B2: end-to-end MATLAB Text2Code export of a
% coupled non-matching application.
%
%   1. Builds the coupled Poisson example (examples/CoupledProblems/Poisson/nonmatching,
%      models 8x16 and 12x24, p = 2, interfacetype = 1) and solves it through the
%      MATLAB path at k+k ranks (phase9_fullsolve('run'), production cmakecompile).
%   2. Exports the same pde/mesh cell arrays with the production exporttext2code.
%   3. Requires both generated pdeapp1.txt and pdeapp2.txt to contain interfacetype = 1.
%   4. Configures and builds the generated coupled bundle (its CMakeLists runs the
%      installed text2code on both models at build time), runs it at k+k ranks (its
%      main runs text2code again to generate datain1/datain2), and requires the donor
%      diagnostics of both directions, Gamma^1 and Gamma^2.
%   5. Compares the bundle solution with the MATLAB solution in the global element order.
%      The bundle's datain comes from text2code, whose partition (and single-rank element
%      order) need not equal the MATLAB one. Each bundle element is identified by its
%      physical geometry: the DG nodes text2code wrote to datain<m>/sol<r>.bin for the
%      owned elements of rank r (the first elements of the rank, in the order of
%      outudg_np<r>.bin) are matched to the global DG nodes of the MATLAB mesh, and every
%      node coordinate must agree to 1e-12 before the solutions are compared.
%
% k = ranks per model (default 1, the B2 acceptance run).
%
% EXASIM_PHASE9_PREFIX (or EXASIM_PHASE6_PREFIX) is the tested installation; its
% bin/text2code must include the Phase 9 interfacetype parser.

if nargin < 1, k = 1; end
if nargin < 2, exampleName = 'nonmatching'; end
if nargin < 3, orientation = 'tri-quad'; end

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
example = fullfile(repo, 'examples', 'CoupledProblems', 'Poisson', exampleName);
addpath(example); addpath(here);
work = fullfile(tempdir, sprintf('exasim_phase9_text2code_%d', k));
if exist(work, 'dir'), rmdir(work, 's'); end
mkdir(work);
old = pwd; cleanup = onCleanup(@() cd(old)); cd(work);
run(fullfile(repo, 'frontends', 'Matlab', 'exasim_setup.m'));
prefix = getenv('EXASIM_PHASE9_PREFIX');
if isempty(prefix), prefix = getenv('EXASIM_PHASE6_PREFIX'); end
assert(~isempty(prefix), 'Set EXASIM_PHASE9_PREFIX to the tested Exasim installation');
setenv('EXASIM_PREFIX', prefix);
text2code = fullfile(prefix, 'bin', 'text2code');
assert(isfile(text2code), 'text2code is missing: %s', text2code);

% --- 1. the coupled example and its MATLAB solution --------------------------------
if strcmp(exampleName, 'mixedmeshes')
    triangleOnModel1 = strcmp(orientation, 'tri-quad');
    assert(triangleOnModel1 || strcmp(orientation, 'quad-tri'), 'unknown mixed orientation');
end
pdeapp1;
pdeapp2;
for m = 1:2
    assert(pde{m}.interfacetype == 1, 'the coupled example must select interfacetype = 1');
    pde{m}.mpiprocs = k;
    pde{m}.datapath = string(fullfile(work, 'matlab'));
    pde{m}.builddir = string(fullfile(work, 'matlab_build'));
    pde{m}.buildpath = pde{m}.builddir;
    elemtype = double(size(mesh{m}.t,1) == 4);
    f = facenumbering(mesh{m}.p, mesh{m}.t, elemtype, mesh{m}.boundaryexpr, []);
    mesh{m}.dgnodes = createdgnodes(mesh{m}.p, mesh{m}.t, f, [], [], pde{m}.porder);
end
mkdir(fullfile(work, 'matlab'));
pdeExport = pde; meshExport = mesh;
ref = phase9_fullsolve('run', pde, mesh);
assert(ref.status == 0, 'MATLAB coupled solve failed');
assert(ref.donor(1) && ref.donor(2), 'MATLAB coupled solve did not run both donor searches');

% --- 2./3. production exporter and the generated pdeapp files --------------------------
dest = fullfile(work, 'bundle');
exporttext2code(pdeExport, meshExport, dest);
for m = 1:2
    app = fileread(fullfile(dest, sprintf('pdeapp%d.txt', m)));
    assert(~isempty(regexp(app, '(^|\n)\s*interfacetype\s*=\s*1\s*;', 'once')), ...
        'exported pdeapp%d.txt does not contain interfacetype = 1', m);
    fprintf('PHASE9 text2code export pdeapp%d.txt: interfacetype = 1\n', m);
end

% --- 4. build and run the generated coupled bundle ------------------------------------
cmakeExe = 'cmake';
[s, ~] = system('command -v cmake'); if s ~= 0, cmakeExe = '/opt/homebrew/bin/cmake'; end
build = fullfile(dest, 'build');
cfg = sprintf(['%s -S %s -B %s -DCMAKE_BUILD_TYPE=Release -DExasim_DIR=%s ' ...
    '-DEXASIM_TEXT2CODE_EXECUTABLE=%s'], cmakeExe, q(dest), q(build), ...
    q(fullfile(prefix, 'lib', 'cmake', 'Exasim')), q(text2code));
run_checked(cfg, 'configure the generated bundle');
run_checked(sprintf('%s --build %s -j 8', cmakeExe, q(build)), 'build the generated bundle');
exe = fullfile(build, 'exasimapp');
assert(isfile(exe), 'the generated bundle did not produce %s', exe);
mpirun = 'mpirun';
mpitxt = fullfile(char(pdeExport{1}.builddir), 'build', 'mpiexec.txt');
if isfile(mpitxt), mpirun = strtrim(fileread(mpitxt)); end
cmd = sprintf('cd %s && %s --oversubscribe -np %d %s pdeapp1.txt %d pdeapp2.txt %d 2>&1', ...
    q(dest), mpirun, 2*k, q(exe), k, k);
[status, logtext] = system(cmd);
fid = fopen(fullfile(work, 'bundle_run.log'), 'w'); fprintf(fid, '%s', logtext); fclose(fid);
assert(status == 0, 'the generated bundle failed (status %d); log in %s', status, ...
    fullfile(work, 'bundle_run.log'));
for d = 1:2
    assert(contains(logtext, sprintf('Non-matching donor search, Gamma^%d', d)), ...
        'the bundle did not run the Gamma^%d donor search', d);
end
fprintf('PHASE9 text2code bundle %d+%d: donor search Gamma^1 and Gamma^2 present\n', k, k);
nwt = regexp(logtext, 'Newton Iteration:\s*[0-9]+,\s*Alpha:[^\n]*Updated Norm:\s*([-+0-9.eE]+)', 'tokens');
if ~isempty(nwt), fprintf('PHASE9 text2code bundle final Newton residual %s\n', nwt{end}{1}); end

% --- 5. bundle solution against the MATLAB solution -----------------------------------
worst = 0;
for m = 1:2
    [s, x] = bundle_solution(dest, m, k);
    ref_dg = meshExport{m}.dgnodes;
    assert(isequal(size(s, 3), size(ref.sol{m}, 3)), 'model %d: bundle and MATLAB element counts differ', m);
    assert(isequal(size(x), size(ref_dg)), 'model %d: bundle and MATLAB geometry sizes differ', m);
    % Associate every bundle element with its global element through physical geometry:
    % the element centroid gives a candidate, and the full nodal coordinates (in node
    % order) must then agree, so a wrong association or a node permutation fails.
    perm = match_elements(x, ref_dg);
    geomErr = max(abs(x(:, :, :) - ref_dg(:, :, perm)), [], 'all');
    assert(geomErr <= 1e-12, 'model %d: bundle element geometry does not match (%.3e)', m, geomErr);
    g = zeros(size(ref.sol{m}));
    g(:, :, perm) = s;
    assert(isequal(size(g), size(ref.sol{m})), 'model %d: bundle and MATLAB solution sizes differ', m);
    rel = norm(g(:) - ref.sol{m}(:)) / norm(ref.sol{m}(:));
    fprintf('PHASE9 text2code model %d: bundle vs MATLAB rel %.3e (%d elements, geometry match %.1e)\n', ...
        m, rel, size(s, 3), geomErr);
    worst = max(worst, rel);
end
assert(worst <= 1e-10, 'bundle and MATLAB solutions differ by %.3e', worst);
fprintf('PASS phase9_text2code %d+%d (max rel %.3e)\n', k, k, worst);
result = struct('worst', worst, 'k', k);
end

function run_checked(cmd, what)
[status, out] = system([cmd ' 2>&1']);
if status ~= 0
    fprintf('%s\n', out);
    error('phase9_text2code: failed to %s (status %d)', what, status);
end
end

function s = q(p)
p = char(p);
assert(~contains(p, ''''), 'Phase 9 test paths may not contain a single quote');
s = ['''' p ''''];
end

function [s, x] = bundle_solution(dest, m, k)
% Owned-element solution and DG nodes of model m, concatenated over the k ranks.
s = []; x = [];
for r = 0:k-1
    [~, ~, n3, ~, block] = read_rank(fullfile(dest, sprintf('dataout%d', m), sprintf('outudg_np%d.bin', r)));
    xr = read_datain_xdg(fullfile(dest, sprintf('datain%d', m)), r);
    assert(size(xr, 3) >= n3, 'datain%d rank %d has fewer elements than its solution', m, r);
    s = cat(3, s, block(:, :, :, end));
    x = cat(3, x, xr(:, :, 1:n3));
end
end

function x = read_datain_xdg(dirname, r)
% DG nodes of rank r from the text2code initial-solution file (writesol): one double
% with the length of nsize, nsize, ndims, then xdg (npe x nd x ne) when nsize(2) > 0.
fname = fullfile(dirname, sprintf('sol%d.bin', r + 1));
if ~isfile(fname), fname = fullfile(dirname, 'sol.bin'); end
fid = fopen(fname, 'r');
assert(fid >= 0, 'cannot open %s', fname);
len = fread(fid, 1, 'double');
nsize = fread(fid, len, 'double');
ndims = fread(fid, nsize(1), 'double');
ne = ndims(1); npe = ndims(4); ncx = ndims(12);
assert(nsize(2) == npe*ncx*ne, 'unexpected xdg size in %s', fname);
x = reshape(fread(fid, nsize(2), 'double'), [npe, ncx, ne]);
fclose(fid);
end

function perm = match_elements(x, ref)
% perm(i) = global element of bundle element i, from centroids (rounded to 1e-9); the
% caller then checks every node coordinate.
cb = squeeze(mean(x, 1)).'; cr = squeeze(mean(ref, 1)).';
if size(x, 3) == 1, cb = cb.'; cr = cr.'; end
[kb, ib] = sortrows(round(cb*1e9));
[kr, ir] = sortrows(round(cr*1e9));
assert(isequal(kb, kr), 'bundle and MATLAB element centroids differ');
assert(size(unique(kr, 'rows'), 1) == size(kr, 1), 'element centroids are not unique');
perm = zeros(size(x, 3), 1);
perm(ib) = ir;
end

function [n1, n2, n3, nsteps, payload] = read_rank(fname)
fid = fopen(fname, 'r');
assert(fid >= 0, 'cannot open %s', fname);
hdr = fread(fid, 3, 'double').';
n1 = hdr(1); n2 = hdr(2); n3 = hdr(3);
payload = fread(fid, 'double');
fclose(fid);
nsteps = numel(payload) / (n1*n2*n3);
assert(nsteps >= 1 && nsteps == round(nsteps), 'bad solution file %s', fname);
payload = reshape(payload, [n1, n2, n3, nsteps]);
end
