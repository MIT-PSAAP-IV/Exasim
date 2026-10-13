function phase9_serialization()
%PHASE9_SERIALIZATION [nonmatching] Phase 9: MATLAB interfacetype field, validation,
% app.bin slot 34, user extras, text2code export key and the parameter-sweep
% template round trip.  Run: matlab -batch "phase9_serialization" from tests/nonmatching.
%
% writeapptemplate is a local function of Postprocessing/exasim.m; it is extracted
% from that file into a scratch function so the production code itself is tested.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
run(fullfile(repo, 'frontends', 'Matlab', 'exasim_setup.m'));
work = tempname;
mkdir(work);
cleanup = onCleanup(@() cleanup_work(work));
failures = {};

pde = base_pde();
failures = check(failures, isfield(pde, 'interfacetype') && pde.interfacetype == 0, ...
    'initializepde default interfacetype is 0');
[~, problem] = write_and_read(pde, work, 'default.bin');
failures = check(failures, numel(problem) >= 35 && problem(35) == 0, ...
    'default interfacetype writes 0 at zero-based slot 34');

% Lower-level public initializer (Revision 1, N2).
app = initializeapp("Backend");
failures = check(failures, isfield(app, 'interfacetype') && app.interfacetype == 0, ...
    'initializeapp default interfacetype is 0');
pde = base_pde();
pde.interfacetype = app.interfacetype;
[~, problem] = write_and_read(pde, work, 'initializeapp.bin');
failures = check(failures, problem(35) == 0, 'initializeapp default writes 0 at slot 34');

pde = rmfield(base_pde(), 'interfacetype');
[~, problem] = write_and_read(pde, work, 'missing.bin');
failures = check(failures, problem(35) == 0, 'absent interfacetype field writes 0 at slot 34');

for value = {0, 1, int32(1)}
    pde = base_pde();
    pde.interfacetype = value{1};
    [~, problem] = write_and_read(pde, work, 'explicit.bin');
    failures = check(failures, problem(35) == double(value{1}), ...
        sprintf('interfacetype=%g written at slot 34', double(value{1})));
end

invalid = {2, -1, 0.5, NaN, true, [0 1], '1', 1i};
for k = 1:numel(invalid)
    pde = base_pde();
    pde.interfacetype = invalid{k};
    filename = fullfile(work, 'invalid.bin');
    if isfile(filename), delete(filename); end
    rejected = false;
    try
        writeapp(pde, filename, 'native');
    catch err
        rejected = contains(err.message, 'interfacetype');
    end
    failures = check(failures, rejected, sprintf('invalid interfacetype #%d is rejected', k));
    failures = check(failures, ~isfile(filename), sprintf('invalid interfacetype #%d writes no app.bin', k));
end

pde = base_pde();
pde.interfacetype = 1;
pde.problem = [7 8 9];
[~, problem] = write_and_read(pde, work, 'extras.bin');
failures = check(failures, numel(problem) == 38, '35 fixed slots plus three user extras');
failures = check(failures, problem(35) == 1 && isequal(problem(36:end).', [7 8 9]), ...
    'user extras follow slot 34 in order');

% Parameter-sweep template round trip.
template = extract_local_function(fullfile(repo, 'frontends', 'Matlab', 'Postprocessing', 'exasim.m'), ...
    'writeapptemplate', work);
pde = base_pde();
pde.uniformrefinementlevel = 1;
pde.problem = [7 8];
pde.flag = 5;
[first, problem1, flag1] = write_and_read(pde, work, 'sweep1.bin');
second = template(first);
[second, problem2, flag2] = write_and_read(second, work, 'sweep2.bin');
failures = check(failures, isequal(problem2, problem1), sprintf( ...
    'sweep template round trip preserves the problem vector (%d -> %d values)', numel(problem1), numel(problem2)));
failures = check(failures, isequal(flag2, flag1), sprintf( ...
    'sweep template round trip preserves the flag vector (%d -> %d values)', numel(flag1), numel(flag2)));
failures = check(failures, isequal(second.factor, first.factor), 'sweep template round trip preserves factor');
failures = check(failures, isequal(second.solversparam, first.solversparam), ...
    'sweep template round trip preserves solversparam');

% text2code export (writeinputfile is the exporter's key writer).
source = fileread(fullfile(repo, 'frontends', 'Matlab', 'Preprocessing', 'writeinputfile.m'));
failures = check(failures, contains(source, '"interfacefluxmap", "interfacetype"'), ...
    'writeinputfile exports interfacetype');

if isempty(failures)
    fprintf('PASS phase9_serialization\n');
else
    fprintf('FAIL phase9_serialization (%d failures)\n', numel(failures));
    error('phase9_serialization failed: %s', strjoin(failures, '; '));
end
end

function cleanup_work(work)
if contains(path, work), rmpath(work); end
rmdir(work, 's');
end

function failures = check(failures, condition, message)
if ~condition
    fprintf('FAIL: %s\n', message);
    failures{end+1} = message;
end
end

function pde = base_pde()
pde = initializepde("Backend");
names = {'nd','ncx','nc','ncu','ncq','ncw','nco','nch','ncp','nce','nsca','nvec','nten','nsurf','nvqoi', ...
    'mpiprocs','ncv','ncs','neb','nfb','elemtype','nodetype','porder','pgauss'};
for k = 1:numel(names)
    if ~isfield(pde, names{k}) || isempty(pde.(names{k})), pde.(names{k}) = 1; end
end
% Fields that preprocessing.m normally derives from the mesh before writeapp.
zeros0 = {'coupledinterface','coupledcondition','coupledboundarycondition','AVdistfunction'};
for k = 1:numel(zeros0)
    if ~isfield(pde, zeros0{k}), pde.(zeros0{k}) = 0; end
end
end

function [app, problem, flag] = write_and_read(pde, work, name)
filename = fullfile(work, name);
app = writeapp(pde, filename, 'native');
fid = fopen(filename, 'r');
numsize = fread(fid, 1, 'double');
nsize = fread(fid, numsize, 'double');
fread(fid, nsize(1), 'double');
flag = fread(fid, nsize(2), 'double');
problem = fread(fid, nsize(3), 'double');
fclose(fid);
end

function handle = extract_local_function(file, name, work)
text = fileread(file);
startIndex = regexp(text, ['function\s+\w+\s*=\s*' name '\s*\('], 'once');
assert(~isempty(startIndex), 'local function %s not found in %s', name, file);
rest = text(startIndex:end);
stop = regexp(rest, '\nend\s*(\n|$)', 'once');
assert(~isempty(stop), 'end of local function %s not found', name);
fname = ['p9_' name];
body = regexprep(rest(1:stop + 3), ['\<' name '\>'], fname, 'once');
fid = fopen(fullfile(work, [fname '.m']), 'w');
fprintf(fid, '%s\n', body);
fclose(fid);
addpath(work);
handle = str2func(fname);
end
