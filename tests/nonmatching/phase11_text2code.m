function result = phase11_text2code(caseName, k, publish)
%PHASE11_TEXT2CODE MATLAB-to-Text2Code round trip of a 3D coupled Poisson example.
%
%   phase11_text2code(caseName, k, publish)
%
%   caseName: matching3d_tet, matching3d_hex, nonmatching3d_tet or
%             nonmatching3d_hex (examples/CoupledProblems/Poisson/<caseName>).
%   k:        ranks per model (default 1).
%   publish:  when true, copy the exported source files into
%             apps/coupledproblems/poisson/<caseName> (default false).
%
%   1. Solve the MATLAB example at k+k ranks (phase9_fullsolve('run')).
%   2. Export the same pde/mesh cells with the production exporttext2code and
%      require the exported interfacetype to match the example.
%   3. Build the exported bundle against EXASIM_PHASE9_PREFIX and run it at k+k
%      ranks. Non-matching cases must print both donor-search diagnostics.
%   4. Match bundle elements to the MATLAB elements by physical DG nodes
%      (<= 1e-12) and require relative solution agreement <= 2e-11, the
%      predeclared cross-frontend criterion.

if nargin < 1, caseName = 'nonmatching3d_tet'; end
if nargin < 2, k = 1; end
if nargin < 3, publish = false; end

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
example = fullfile(repo, 'examples', 'CoupledProblems', 'Poisson', caseName);
assert(isfolder(example), 'unknown 3D example %s', caseName);
addpath(example);
addpath(here);
work = fullfile(tempdir, sprintf('exasim_phase11_text2code_%s_%d', caseName, k));
if exist(work, 'dir'), rmdir(work, 's'); end
mkdir(work);
old = pwd; cleanup = onCleanup(@() cd(old)); cd(work);
run(fullfile(repo, 'frontends', 'Matlab', 'exasim_setup.m'));
prefix = getenv('EXASIM_PHASE9_PREFIX');
if isempty(prefix), prefix = fullfile(fileparts(repo), 'Exasim-build', 'install'); end
assert(isfile(fullfile(prefix, 'lib', 'cmake', 'Exasim', 'ExasimConfig.cmake')), ...
    'Phase 11 requires an installed Exasim prefix');
setenv('EXASIM_PREFIX', prefix); setenv('EXASIM_PHASE9_PREFIX', prefix);
text2code = fullfile(prefix, 'bin', 'text2code');
assert(isfile(text2code), 'text2code is missing: %s', text2code);

% --- 1. MATLAB solution -----------------------------------------------------------
pde = cell(2,1); mesh = cell(2,1);
pdeapp1;
pdeapp2;
itype = pde{1}.interfacetype;
assert(pde{2}.interfacetype == itype, 'the two models select different interface types');
for m = 1:2
    pde{m}.mpiprocs = k;
    pde{m}.datapath = string(fullfile(work, 'matlab'));
    pde{m}.builddir = string(fullfile(work, 'matlab_build'));
    pde{m}.buildpath = pde{m}.builddir;
    elemtype = double(size(mesh{m}.t,1) == 8);
    f = facenumbering(mesh{m}.p, mesh{m}.t, elemtype, mesh{m}.boundaryexpr, []);
    mesh{m}.dgnodes = createdgnodes(mesh{m}.p, mesh{m}.t, f, [], [], pde{m}.porder);
end
mkdir(fullfile(work, 'matlab'));
% Export the straight-sided meshes without DG nodes: Text2Code generates them
% from grid<m>.bin, so the bundle needs no xdg<m>.bin. The DG nodes are kept
% separately for the element matching below.
pdeExport = pde; meshExport = mesh; refDgnodes = cell(2,1);
for m = 1:2
    refDgnodes{m} = mesh{m}.dgnodes;
    meshExport{m} = rmfield(meshExport{m}, 'dgnodes');
end
ref = phase9_fullsolve('run', pde, mesh);
assert(ref.status == 0, 'MATLAB coupled solve failed');
if itype == 1
    assert(ref.donor(1) && ref.donor(2), 'MATLAB solve did not run both donor searches');
end

% --- 2. production exporter -------------------------------------------------------
dest = fullfile(work, 'bundle');
exporttext2code(pdeExport, meshExport, dest);
for m = 1:2
    app = fileread(fullfile(dest, sprintf('pdeapp%d.txt', m)));
    assert(~isempty(regexp(app, sprintf('(^|\\n)\\s*interfacetype\\s*=\\s*%d\\s*;', itype), 'once')), ...
        'exported pdeapp%d.txt does not contain interfacetype = %d', m, itype);
end

% --- 3. build and run the bundle -------------------------------------------------
cmakeExe = 'cmake';
[s, ~] = system('command -v cmake'); if s ~= 0, cmakeExe = '/opt/homebrew/bin/cmake'; end
build = fullfile(work, 'bundle_build');
run_checked(sprintf(['%s -S %s -B %s -DCMAKE_BUILD_TYPE=Release -DExasim_DIR=%s ' ...
    '-DEXASIM_TEXT2CODE_EXECUTABLE=%s'], cmakeExe, q(dest), q(build), ...
    q(fullfile(prefix, 'lib', 'cmake', 'Exasim')), q(text2code)), 'configure the bundle');
run_checked(sprintf('%s --build %s -j 8', cmakeExe, q(build)), 'build the bundle');
exe = fullfile(build, 'exasimapp');
assert(isfile(exe), 'the bundle did not produce %s', exe);
mpirun = 'mpirun';
mpitxt = fullfile(char(pdeExport{1}.builddir), 'build', 'mpiexec.txt');
if isfile(mpitxt), mpirun = strtrim(fileread(mpitxt)); end
[status, logtext] = system(sprintf('cd %s && %s --oversubscribe -np %d %s pdeapp1.txt %d pdeapp2.txt %d 2>&1', ...
    q(dest), mpirun, 2*k, q(exe), k, k));
fid = fopen(fullfile(work, 'bundle_run.log'), 'w'); fprintf(fid, '%s', logtext); fclose(fid);
assert(status == 0, 'the bundle failed (status %d); log in %s', status, fullfile(work, 'bundle_run.log'));
if itype == 1
    for d = 1:2
        assert(contains(logtext, sprintf('Non-matching donor search, Gamma^%d', d)), ...
            'the bundle did not run the Gamma^%d donor search', d);
    end
end
nwt = regexp(logtext, 'Updated Norm:\s*([-+0-9.eE]+)', 'tokens');

% --- 4. compare ----------------------------------------------------------------
worst = 0;
for m = 1:2
    [sb, xb] = bundle_solution(dest, m, k);
    xr = refDgnodes{m};
    perm = match_elements(xb, xr);
    geomErr = max(abs(xb - xr(:,:,perm)), [], 'all');
    assert(geomErr <= 1e-12, 'model %d: bundle geometry mismatch %.3e', m, geomErr);
    g = zeros(size(ref.sol{m})); g(:,:,perm) = sb;
    rel = norm(g(:) - ref.sol{m}(:)) / norm(ref.sol{m}(:));
    fprintf('PHASE11 text2code %s model %d: bundle vs MATLAB rel %.3e (%d elements)\n', ...
        caseName, m, rel, size(sb,3));
    worst = max(worst, rel);
end
assert(worst <= 2e-11, 'bundle and MATLAB solutions differ by %.3e', worst);
if ~isempty(nwt), fprintf('PHASE11 text2code %s final Newton residual %s\n', caseName, nwt{end}{1}); end

if publish
    app = fullfile(repo, 'apps', 'coupledproblems', 'poisson', caseName);
    if ~isfolder(app), mkdir(app); end
    for name = {'pdeapp1.txt','pdeapp2.txt','pdemodel1.txt','pdemodel2.txt','grid1.bin','grid2.bin'}
        copyfile(fullfile(dest, name{1}), fullfile(app, name{1}));
    end
    % The published apps use the same RunText2CodeApp runtime as the other
    % coupled Poisson apps instead of the exporter's stand-alone main.
    standard = fullfile(repo, 'apps', 'coupledproblems', 'poisson', 'nonmatching');
    copyfile(fullfile(standard, 'main.cpp'), fullfile(app, 'main.cpp'));
    copyfile(fullfile(standard, 'CMakeLists.txt'), fullfile(app, 'CMakeLists.txt'));
    fprintf('PHASE11 text2code %s: published to %s\n', caseName, app);
end
fprintf('PASS phase11_text2code %s %d+%d (max rel %.3e)\n', caseName, k, k, worst);
result = struct('worst', worst, 'k', k, 'case', caseName);
end

function run_checked(cmd, what)
[status, out] = system([cmd ' 2>&1']);
if status ~= 0
    fprintf('%s\n', out);
    error('phase11_text2code: failed to %s (status %d)', what, status);
end
end

function s = q(p)
p = char(p);
assert(~contains(p, ''''), 'test paths may not contain a single quote');
s = ['''' p ''''];
end

function [s, x] = bundle_solution(dest, m, k)
% Owned-element solution and DG nodes of model m, concatenated over the k ranks.
s = []; x = [];
for r = 0:k-1
    fid = fopen(fullfile(dest, sprintf('dataout%d', m), sprintf('outudg_np%d.bin', r)), 'r');
    hdr = fread(fid, 3, 'double').'; payload = fread(fid, 'double'); fclose(fid);
    nsteps = numel(payload)/prod(hdr);
    block = reshape(payload, [hdr nsteps]);
    xr = read_datain_xdg(fullfile(dest, sprintf('datain%d', m)), r);
    s = cat(3, s, block(:,:,:,end));
    x = cat(3, x, xr(:,:,1:hdr(3)));
end
end

function x = read_datain_xdg(dirname, r)
fname = fullfile(dirname, sprintf('sol%d.bin', r + 1));
if ~isfile(fname), fname = fullfile(dirname, 'sol.bin'); end
fid = fopen(fname, 'r');
len = fread(fid, 1, 'double'); nsize = fread(fid, len, 'double');
nd = fread(fid, nsize(1), 'double');
ne = nd(1); npe = nd(4); ncx = nd(12);
assert(nsize(2) == npe*ncx*ne, 'unexpected xdg size in %s', fname);
x = reshape(fread(fid, nsize(2), 'double'), [npe, ncx, ne]);
fclose(fid);
end

function perm = match_elements(x, ref)
cb = squeeze(mean(x, 1)).'; cr = squeeze(mean(ref, 1)).';
[kb, ib] = sortrows(round(cb*1e9));
[kr, ir] = sortrows(round(cr*1e9));
assert(isequal(kb, kr), 'bundle and MATLAB element centroids differ');
assert(size(unique(kr, 'rows'), 1) == size(kr, 1), 'element centroids are not unique');
perm = zeros(size(x, 3), 1); perm(ib) = ir;
end
