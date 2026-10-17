function result = phase11_frontends(caseName, frontends)
%PHASE11_FRONTENDS MATLAB/Python/Julia agreement of a 3D coupled Poisson example.
%
%   phase11_frontends(caseName, frontends)
%
%   Runs examples/CoupledProblems/Poisson/<caseName>/pdeapp in each requested
%   frontend (default {'matlab','python','julia'}). Each run writes sol<m>.bin
%   and xdg<m>.bin in global element order to <frontend>_run/results. Every
%   run is compared with the MATLAB run: elements are matched by their DG nodes
%   (<= 1e-12), and the relative solution difference must be <= 2e-11, the
%   predeclared cross-frontend criterion.
%
%   Environment: EXASIM_PHASE9_PREFIX (default <repo>/../Exasim-build/install).
%   EXASIM_PYTHON selects the Python interpreter (default /usr/bin/python3; it
%   needs numpy, scipy and sympy). EXASIM_JULIA selects julia (default julia).

if nargin < 1, caseName = 'nonmatching3d_tet'; end
if nargin < 2, frontends = {'matlab','python','julia'}; end
here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
example = fullfile(repo, 'examples', 'CoupledProblems', 'Poisson', caseName);
assert(isfolder(example), 'unknown 3D example %s', caseName);
prefix = getenv('EXASIM_PHASE9_PREFIX');
if isempty(prefix), prefix = fullfile(fileparts(repo), 'Exasim-build', 'install'); end
setenv('EXASIM_PREFIX', prefix);
python = getenv('EXASIM_PYTHON'); if isempty(python), python = '/usr/bin/python3'; end
julia = getenv('EXASIM_JULIA'); if isempty(julia), julia = 'julia'; end

for i = 1:numel(frontends)
    switch frontends{i}
        case 'matlab'
            work = fullfile(tempdir, sprintf('exasim_phase11_frontends_%s', caseName));
            if exist(work, 'dir'), rmdir(work, 's'); end
            mkdir(work); old = pwd; c = onCleanup(@() cd(old)); cd(work);
            evalc('run(fullfile(example, ''pdeapp.m''))');
            clear c;
        case 'python'
            run_checked(sprintf('cd %s && EXASIM_PREFIX=%s %s pdeapp.py', q(example), q(prefix), q(python)), 'Python');
        case 'julia'
            run_checked(sprintf('cd %s && EXASIM_PREFIX=%s %s --project=%s pdeapp.jl', q(example), q(prefix), ...
                q(julia), q(fullfile(repo, 'frontends', 'Julia', 'Exasim'))), 'Julia');
        otherwise
            error('unknown frontend %s', frontends{i});
    end
end

ref = load_results(fullfile(example, 'matlab_run', 'results'));
worst = 0;
for i = 1:numel(frontends)
    if strcmp(frontends{i}, 'matlab'), continue; end
    other = load_results(fullfile(example, [frontends{i} '_run'], 'results'));
    for m = 1:2
        perm = match_elements(other.xdg{m}, ref.xdg{m});
        geomErr = max(abs(other.xdg{m} - ref.xdg{m}(:,:,perm)), [], 'all');
        assert(geomErr <= 1e-12, '%s model %d: DG nodes differ (%.3e)', frontends{i}, m, geomErr);
        g = zeros(size(ref.sol{m})); g(:,:,perm) = other.sol{m};
        rel = norm(g(:) - ref.sol{m}(:)) / norm(ref.sol{m}(:));
        fprintf('PHASE11 frontends %s %s model %d: vs MATLAB rel %.3e\n', caseName, frontends{i}, m, rel);
        worst = max(worst, rel);
    end
end
assert(worst <= 2e-11, 'frontends differ by %.3e', worst);
fprintf('PASS phase11_frontends %s (max rel %.3e)\n', caseName, worst);
result = struct('case', caseName, 'worst', worst);
end

function r = load_results(dirname)
r.sol = cell(2,1); r.xdg = cell(2,1);
for m = 1:2
    r.sol{m} = readbin(fullfile(dirname, sprintf('sol%d.bin', m)));
    r.xdg{m} = readbin(fullfile(dirname, sprintf('xdg%d.bin', m)));
end
end

function a = readbin(filename)
fid = fopen(filename, 'r');
assert(fid >= 0, 'missing result file %s', filename);
nd = fread(fid, 1, 'double'); dims = fread(fid, nd, 'double').';
a = reshape(fread(fid, prod(dims), 'double'), [dims 1]);
fclose(fid);
end

function perm = match_elements(x, ref)
cb = squeeze(mean(x, 1)).'; cr = squeeze(mean(ref, 1)).';
[kb, ib] = sortrows(round(cb*1e9));
[kr, ir] = sortrows(round(cr*1e9));
assert(isequal(kb, kr), 'element centroids differ between frontends');
perm = zeros(size(x, 3), 1); perm(ib) = ir;
end

function run_checked(cmd, what)
[status, out] = system([cmd ' 2>&1']);
if status ~= 0
    fprintf('%s\n', out);
    error('phase11_frontends: %s run failed (status %d)', what, status);
end
m = regexp(out, 'Maximum nodal error[^\n]*', 'match');
if ~isempty(m), fprintf('PHASE11 frontends %s: %s\n', what, m{end}); end
end

function s = q(p)
p = char(p);
s = ['''' p ''''];
end
