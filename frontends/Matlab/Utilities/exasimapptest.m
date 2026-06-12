function results = exasimapptest(np)
%EXASIMAPPTEST Cross-frontend and built-in ABI smoke tests for Exasim.
%
% RESULTS = EXASIMAPPTEST() exercises one Poisson problem through:
%   1. Matlab, Julia, and Python frontends in examples/Poisson/poisson2d.
%   2. BuiltIn dynamic-library ABI executable in apps/builtinlibrary for
%      apps/poisson/poisson2d.
%
% Optional environment overrides:
%   EXASIM_CMAKE   path to cmake
%   EXASIM_MPIRUN  path to mpirun
%   EXASIM_JULIA   path to julia
%   EXASIM_PYTHON  path to python3

testFile = mfilename('fullpath');
testDir = fileparts(testFile);
exasimDir = findExasimRoot(testDir);

exampleDir = fullfile(exasimDir, 'examples', 'Poisson', 'poisson2d');
exampleBuildDir = fullfile(exasimDir, 'examples', 'build');
builtinAppDir = fullfile(exasimDir, 'apps', 'builtinlibrary');
builtinAppBuildDir = fullfile(builtinAppDir, 'build');
builtinInputFile = '../poisson/poisson2d/pdeapp.txt';
builtinAppExe = fullfile(builtinAppBuildDir, 'exasimapp');
builtinExasimDir = fullfile(char(exasim_install_prefix()), 'lib', 'cmake', 'Exasim');

cmake = getenvDefault('EXASIM_CMAKE', findExecutable('cmake', '/opt/homebrew/bin/cmake'));
mpirun = getenvDefault('EXASIM_MPIRUN', findExecutable('mpirun', '/opt/homebrew/bin/mpirun'));
julia = getenvDefault('EXASIM_JULIA', findJuliaExecutable());
python = getenvDefault('EXASIM_PYTHON', findPythonExecutable());
juliaProjectDir = fullfile(exasimDir, 'frontends', 'Julia', 'Exasim');
paraview = findParaviewExecutable();
pathPrefix = pathEnvironmentPrefix(paraview);
if isnan(np) || np < 1
    error('np must be a positive integer.');
end
np = round(np);

results = struct('name', {}, 'command', {}, 'status', {});

fprintf('==> Exasim smoke tests\n');
fprintf('Exasim root: %s\n', exasimDir);
fprintf('Example build path: %s\n', exampleBuildDir);
fprintf('Builtin app build: %s\n', builtinAppBuildDir);
fprintf('Builtin Exasim_DIR: %s\n', builtinExasimDir);
fprintf('MPI ranks  : %d\n\n', np);

%% Frontend tests
resetCMakeCache(exampleBuildDir);
results(end+1) = runMatlabScript('frontend-matlab-poisson2d', exampleDir, 'pdeapp.m'); %#ok<SAGROW>
resetCMakeCache(exampleBuildDir);

juliaCmd = sprintf('%s%s --project=%s -e %s', ...
    pathPrefix, ...
    shellQuote(julia), ...
    shellQuote(juliaProjectDir), ...
    shellQuote(sprintf('using Pkg; Pkg.instantiate(); include(%s)', ...
        quotedForJulia(fullfile(exampleDir, 'pdeapp.jl')))));
results(end+1) = runShellTest('frontend-julia-poisson2d', exampleDir, juliaCmd); %#ok<SAGROW>

resetCMakeCache(exampleBuildDir);
results(end+1) = runShellTest('frontend-python-poisson2d', exampleDir, ...
    sprintf('%sPYTHONPATH=%s %s %s', ...
        pathPrefix, ...
        shellQuote(fullfile(exasimDir, 'frontends', 'Python')), ...
        shellQuote(python), ...
        shellQuote(fullfile(exampleDir, 'pdeapp.py')))); %#ok<SAGROW>

%% BuiltIn dynamic-library ABI executable
resetCMakeCache(builtinAppBuildDir);
cmakeConfigure = strjoin({ ...
    shellQuote(cmake), ...
    '-S', shellQuote(builtinAppDir), ...
    '-B', shellQuote(builtinAppBuildDir), ...
    '-D', ['Exasim_DIR=' shellQuote(builtinExasimDir)], ...
    '-D', 'EXASIM_MPI=ON'}, ' ');
results(end+1) = runShellTest('builtinlibrary-cmake-configure', exasimDir, cmakeConfigure); %#ok<SAGROW>

cmakeBuild = sprintf('%s --build %s --target exasimapp', ...
    shellQuote(cmake), shellQuote(builtinAppBuildDir));
results(end+1) = runShellTest('builtinlibrary-cmake-build', exasimDir, cmakeBuild); %#ok<SAGROW>

runBuiltin = sprintf('%s -np %d %s %s', ...
    shellQuote(mpirun), np, shellQuote(builtinAppExe), shellQuote(builtinInputFile));
results(end+1) = runShellTest('builtinlibrary-run-poisson2d', builtinAppDir, runBuiltin); %#ok<SAGROW>

%% Summary
fprintf('\n==> Exasim test summary\n');
failed = false;
for i = 1:numel(results)
    if results(i).status == 0
        mark = 'PASS';
    else
        mark = 'FAIL';
        failed = true;
    end
    fprintf('%-6s %s\n', mark, results(i).name);
end

if failed
    error('One or more Exasim smoke tests failed.');
end

fprintf('All Exasim smoke tests passed.\n');
end

function result = runMatlabScript(name, workDir, scriptName)
    fprintf('\n==> %s\n', name);
    fprintf('cd %s\n', workDir);
    fprintf('run %s\n', scriptName);

    oldDir = pwd();
    cleanup = onCleanup(@() cd(oldDir));
    result = struct('name', name, 'command', ['run ' scriptName], 'status', 1);
    try
        cd(workDir);
        run(scriptName);
        result.status = 0;
    catch err
        fprintf(2, 'FAILED: %s\n', name);
        fprintf(2, '%s\n', getReport(err, 'extended', 'hyperlinks', 'off'));
        result.status = 1;
    end
end

function result = runShellTest(name, workDir, command)
    fprintf('\n==> %s\n', name);
    fprintf('cd %s\n', workDir);
    fprintf('%s\n', command);

    oldDir = pwd();
    cleanup = onCleanup(@() cd(oldDir));
    cd(workDir);
    status = system(command);
    result = struct('name', name, 'command', command, 'status', status);
end

function value = getenvDefault(name, defaultValue)
    value = getenv(name);
    if isempty(value)
        value = defaultValue;
    end
end

function exe = findExecutable(name, fallback)
    [status, output] = system(sprintf('command -v %s', shellQuote(name)));
    if status == 0
        exe = strtrim(output);
    elseif exist(fallback, 'file')
        exe = fallback;
    else
        exe = name;
    end
end

function exe = findJuliaExecutable()
    [status, output] = system('command -v julia');
    if status == 0
        exe = strtrim(output);
        return;
    end

    candidates = {};
    homeDir = getenv('HOME');
    if ~isempty(homeDir)
        candidates{end+1} = fullfile(homeDir, '.juliaup', 'bin', 'julia'); %#ok<AGROW>
        candidates{end+1} = fullfile(homeDir, 'bin', 'julia'); %#ok<AGROW>
    end
    candidates{end+1} = '/opt/homebrew/bin/julia';
    candidates{end+1} = '/usr/local/bin/julia';

    appCandidates = dir('/Applications/Julia*.app/Contents/Resources/julia/bin/julia');
    for i = 1:numel(appCandidates)
        candidates{end+1} = fullfile(appCandidates(i).folder, appCandidates(i).name); %#ok<AGROW>
    end

    for i = 1:numel(candidates)
        if exist(candidates{i}, 'file')
            exe = candidates{i};
            return;
        end
    end

    exe = 'julia';
end

function exe = findPythonExecutable()
    candidates = {'python3', 'python'};
    for i = 1:numel(candidates)
        candidate = candidates{i};
        status = system(sprintf('%s -c %s', ...
            shellQuote(candidate), shellQuote('import numpy')));
        if status == 0
            exe = candidate;
            return;
        end
    end

    for i = 1:numel(candidates)
        candidate = candidates{i};
        [status, output] = system(sprintf('command -v %s', shellQuote(candidate))); %#ok<ASGLU>
        if status == 0
            exe = strtrim(output);
            return;
        end
    end

    exe = 'python3';
end

function exe = findParaviewExecutable()
    candidates = {};
    appCandidates = dir('/Applications/ParaView*.app/Contents/MacOS/paraview');
    for i = 1:numel(appCandidates)
        candidates{end+1} = fullfile(appCandidates(i).folder, appCandidates(i).name); %#ok<AGROW>
    end
    candidates{end+1} = findExecutable('paraview', '/opt/homebrew/bin/paraview'); %#ok<AGROW>

    for i = 1:numel(candidates)
        if exist(candidates{i}, 'file')
            exe = candidates{i};
            return;
        end
    end

    exe = '';
end

function prefix = pathEnvironmentPrefix(exePath)
    if isempty(exePath)
        prefix = '';
        return;
    end

    prefix = sprintf('PATH=%s:$PATH ', shellQuote(fileparts(exePath)));
end

function q = shellQuote(value)
    value = char(value);
    q = ['''' strrep(value, '''', '''"''"''') ''''];
end

function q = quotedForJulia(value)
    value = char(value);
    q = ['"' strrep(value, '"', '\"') '"'];
end

function resetCMakeCache(buildDir)
    if ~exist(buildDir, 'dir')
        return;
    end

    entries = { ...
        'CMakeCache.txt', ...
        'CMakeFiles', ...
        'Makefile', ...
        'cmake_install.cmake', ...
        'build.ninja', ...
        'rules.ninja', ...
        '.ninja_deps', ...
        '.ninja_log'};

    for i = 1:numel(entries)
        pathValue = fullfile(buildDir, entries{i});
        if exist(pathValue, 'dir')
            rmdir(pathValue, 's');
        elseif exist(pathValue, 'file')
            delete(pathValue);
        end
    end
end

function exasimDir = findExasimRoot(startDir)
    exasimDir = char(startDir);
    while true
        if exist(fullfile(exasimDir, 'CMakeLists.txt'), 'file') && ...
           exist(fullfile(exasimDir, 'examples'), 'dir') && ...
           exist(fullfile(exasimDir, 'apps', 'builtinlibrary', 'CMakeLists.txt'), 'file') && ...
           exist(fullfile(exasimDir, 'frontends', 'Matlab'), 'dir') && ...
           exist(fullfile(exasimDir, 'install', 'CMakeLists.txt'), 'file')
            return;
        end

        parentDir = fileparts(exasimDir);
        if strcmp(parentDir, exasimDir)
            error('Could not determine the Exasim root directory from %s.', startDir);
        end
        exasimDir = parentDir;
    end
end
