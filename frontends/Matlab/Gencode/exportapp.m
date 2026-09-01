function dest = exportapp(pde, dest, build)
% Package a relocatable "data transfer app" bundle.
%
% Collects everything the solver needs to run on any machine that has an
% Exasim install: the binary inputs (datain/, plus an empty dataout/), the
% generated kernel .cpp set, a relocatable CMakeLists.txt, main.cpp, a
% run.sh, a manifest, and a copy of the source model. The bundle depends on
% an Exasim install at run time (find_package(Exasim)) but carries no
% absolute paths, so it can be copied anywhere.
%
% Boundary conditions need no text form here: preprocessing has already
% resolved them into the binary datain. Requires preprocessing (datain) and
% kkgencode (kernels) to have run already.
%
% With build=true (default) the bundle is configured, built, and run in a
% throwaway scratch dir against the local Exasim install to verify it works
% before hand-off, leaving the shipped bundle pristine (no build/, empty
% dataout/). Returns the absolute bundle path.

if nargin < 2 || isempty(dest)
    if isfield(pde, 'exportapp') && ~isempty(pde.exportapp)
        dest = pde.exportapp;
    else
        error("exportapp: no destination (set pde.exportapp or pass dest).");
    end
end
if nargin < 3 || isempty(build)
    build = true;
end
dest = string(absolutepath_str(dest));

builddir = pde.builddir;
kernels = builddir + "/kernels";
if ~exist(char(kernels), 'dir')
    error("No generated kernels at %s; run kkgencode(pde) first.", kernels);
end

datain = pde.datapath + "/datain";
if ~exist(char(datain), 'dir')
    error("No datain at %s; run preprocessing(pde, mesh) first.", datain);
end

fprintf("Export Exasim data-transfer app to %s ...\n", dest);
if ~exist(char(dest), 'dir'), mkdir(char(dest)); end

% 1. Runtime inputs + the output directory the solver writes into.
copytree(datain, dest + "/datain");
if ~exist(char(dest + "/dataout"), 'dir'), mkdir(char(dest + "/dataout")); end
process_materialdatabase(pde, dest + "/datain");
write_physicsparamcases_if_needed(pde, dest + "/datain");

% 2. The generated kernel .cpp set.
copytree(kernels, dest + "/kernels");

% 3 + 4. Relocatable CMakeLists.txt + main.cpp from the shared templates.
% KERNEL_DIR resolves inside the bundle at configure time (not an absolute
% path), which is what makes the bundle relocatable. The Exasim install is
% the only external dependency, supplied on the target via Exasim_DIR.
if pde.platform == "gpu"
    if pde.mpiprocs > 1, variant = "gpumpi"; else, variant = "gpu"; end
else
    if pde.mpiprocs > 1, variant = "cpumpi"; else, variant = "cpu"; end
end

prefix = exasim_install_prefix();
numpde = 1;
tmpl = prefix + "/lib/cmake/Exasim/frontend-app";

% 4. Render the CMakeLists.txt and main.cpp templates with the substitutions
if isfield(pde,'frontendprovider') && ~isempty(pde.frontendprovider) && pde.frontendprovider == true
    rendertemplate(tmpl + "/exasimfeapp.cpp.in", dest + "/exasimfeapp.cpp", []);
    rendertemplate(tmpl + "/frontendprovider.cpp.in", dest + "/frontendprovider.cpp", []);
    rendertemplate(tmpl + "/CMakeListsFrontendProvider.txt.in", dest + "/CMakeLists.txt", []);
else
    subs = {"EXASIM_VARIANT", variant; ...
            "MODEL_ID", string(resolve_modelid(pde)); ...
            "KERNEL_DIR", "${CMAKE_CURRENT_SOURCE_DIR}/kernels"};
    rendertemplate(tmpl + "/CMakeLists.txt.in", dest + "/CMakeLists.txt", subs);
    rendertemplate(tmpl + "/main.cpp.in", dest + "/main.cpp", subs);
end

% 5. A copy of the source model, for reference / reproducibility.
modelsrc = string(pde.modelfile) + ".m";
if exist(char(modelsrc), 'file') == 2
    [~, mname, mext] = fileparts(char(modelsrc));
    copyfile(char(modelsrc), char(dest + "/" + mname + mext));
end

% 5b. A text2code DSL file (pdemodel.txt) emitted from the symbolic model, so
% the bundle can regenerate kernels via text2code without the MATLAB frontend.
% Additive: if generation fails, warn and continue (the kernels are already
% shipped in kernels/), mirroring the Python frontend's non-fatal 5b block.
try
    genpdemodel(pde, dest + "/pdemodel.txt");
catch exc
    fprintf("WARNING: pdemodel.txt generation skipped (%s); the exported kernels/ are unaffected.\n", exc.message);
end

% 6. run.sh + manifest with the provenance needed to rebuild elsewhere.
write_runscript(dest, pde, variant, numpde);
write_manifest(dest, pde, variant, numpde, prefix);
write_readme(dest, variant, numpde);

if build
    build_and_run(dest, pde, numpde, prefix);
end

fprintf("Exported data-transfer app: %s\n", dest);
end

% ---------------------------------------------------------------------------

function write_physicsparamcases_if_needed(pde, datain)
if ~isfield(pde, 'physicsparamsweep') || isempty(pde.physicsparamsweep)
    return;
end
cases = export_physicsparamsweepcases(pde.physicsparamsweep, numel(pde.physicsparam));
if size(cases,1) <= 1
    return;
end
fn = char(datain + "/physicsparamcases.bin");
fid = fopen(fn, 'w');
if fid < 0
    error("Unable to write %s.", fn);
end
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, [size(cases,1), size(cases,2)], 'double');
fwrite(fid, cases.', 'double');
end

function cases = export_physicsparamsweepcases(spec, nparam)
if isnumeric(spec)
    if isvector(spec) && nparam == 1
        cases = spec(:);
    else
        cases = spec;
    end
elseif iscell(spec)
    cases = zeros(numel(spec), nparam);
    for i = 1:numel(spec)
        v = spec{i};
        if numel(v) ~= nparam
            error("physicsparamsweep case %d has %d parameters; expected %d.", i, numel(v), nparam);
        end
        cases(i,:) = reshape(v, 1, []);
    end
elseif isstruct(spec)
    if isfield(spec, 'samples')
        cases = export_physicsparamsweepcases(spec.samples, nparam);
    elseif isfield(spec, 'values')
        cases = export_physicsparamsweepcases(spec.values, nparam);
    elseif isfield(spec, 'grid')
        if ~iscell(spec.grid) || numel(spec.grid) ~= nparam
            error("physicsparamsweep.grid must be a cell array with one value vector per physics parameter.");
        end
        grids = cellfun(@(v) v(:), spec.grid, 'UniformOutput', false);
        [meshgrids{1:nparam}] = ndgrid(grids{:});
        cases = zeros(numel(meshgrids{1}), nparam);
        for j = 1:nparam
            cases(:,j) = meshgrids{j}(:);
        end
    else
        error("physicsparamsweep struct must contain samples, values, or grid.");
    end
else
    error("physicsparamsweep must be numeric, a cell array, or a struct.");
end
if size(cases,2) ~= nparam
    error("Each physicsparamsweep row must contain %d physics parameters.", nparam);
end
if any(~isfinite(cases(:)))
    error("physicsparamsweep cases must contain finite numeric values.");
end
end

function write_runscript(dest, pde, variant, numpde)
runline = run_command(pde, numpde, '"$here/build/exasimapp"', ...
                      '"$here/datain/"', '"$here/dataout/out"');
text = [ ...
"#!/usr/bin/env bash"; ...
"# Build and run this exported Exasim data-transfer app."; ...
"# Requires an Exasim install on this machine; point EXASIM_ROOT at its prefix"; ...
"# (the directory that contains lib/cmake/Exasim/). For MPI variants, set"; ...
"# MPIRUN if your launcher is not ""mpirun""."; ...
"#"; ...
"# The bundle was exported for variant """ + variant + """, but the kernels and"; ...
"# datain are arch-independent: retarget to whatever the build machine has by"; ...
"# setting EXASIM_VARIANT (cpu | cpumpi | gpu | gpumpi), e.g."; ...
"#   EXASIM_ROOT=/path/to/install EXASIM_VARIANT=gpu ./run.sh"; ...
"set -euo pipefail"; ...
"here=""$(cd ""$(dirname ""${BASH_SOURCE[0]}"")"" && pwd)"""; ...
": ""${EXASIM_ROOT:?Set EXASIM_ROOT to your Exasim install prefix}"""; ...
"cmake -S ""$here"" -B ""$here/build"" \"; ...
"  -DExasim_DIR=""$EXASIM_ROOT/lib/cmake/Exasim"" \"; ...
"  -DEXASIM_VARIANT=""${EXASIM_VARIANT:-" + variant + "}"""; ...
"cmake --build ""$here/build"" --parallel"; ...
runline; ...
""];
path = char(dest + "/run.sh");
fid = fopen(path, 'w');
fwrite(fid, char(strjoin(text, newline)));
fclose(fid);
fileattrib(path, '+x');
end

function cmd = run_command(pde, numpde, exe, datain, dataout)
% The command line the bundle uses to launch the solver (MPI-aware).
if pde.mpiprocs > 1
    cmd = "${MPIRUN:-mpirun} -np " + string(pde.mpiprocs) + " " + ...
          exe + " " + string(numpde) + " " + datain + " " + dataout;
else
    cmd = exe + " " + string(numpde) + " " + datain + " " + dataout;
end
end

function write_manifest(dest, pde, variant, numpde, prefix)
m = struct();
m.format = "exasim-data-transfer-app/1";
m.frontend = "matlab";
m.modelid = pde.modelid;
m.variant = variant;
m.platform = string(pde.platform);
m.mpiprocs = pde.mpiprocs;
m.numpde = numpde;
if isfield(pde, 'hybrid'), m.hybrid = pde.hybrid; else, m.hybrid = 0; end
if isfield(pde, 'porder'), m.porder = pde.porder; else, m.porder = []; end
m.built_against_prefix = string(prefix);
text = jsonencode(m, 'PrettyPrint', true);
fid = fopen(char(dest + "/manifest.json"), 'w');
fwrite(fid, [char(text) newline]);
fclose(fid);
end

function write_readme(dest, variant, numpde)
text = [ ...
"# Exasim data-transfer app"; ...
""; ...
"Self-contained bundle for the generated model (variant `" + variant + "`). It carries"; ...
"the binary inputs, the generated kernels, and a relocatable CMake build, but"; ...
"relies on an **Exasim install** on the machine where you build it."; ...
""; ...
"## Contents"; ...
"- `datain/`    binary solver inputs (mesh, master, app, solution); boundary"; ...
"               conditions are already resolved here."; ...
"- `dataout/`   solver writes outputs here."; ...
"- `kernels/`   generated kernel .cpp set for this model."; ...
"- `CMakeLists.txt`, `main.cpp`   relocatable build of `exasimapp`."; ...
"- `run.sh`     build + run helper."; ...
"- `manifest.json`   provenance (model id, variant, process count, ...)."; ...
""; ...
"## Build & run"; ...
"```sh"; ...
"EXASIM_ROOT=/path/to/exasim/install ./run.sh"; ...
"```"; ...
"or manually:"; ...
"```sh"; ...
"cmake -S . -B build -DExasim_DIR=$EXASIM_ROOT/lib/cmake/Exasim"; ...
"cmake --build build --parallel"; ...
"./build/exasimapp " + string(numpde) + " datain/ dataout/out"; ...
"```"; ...
""];
fid = fopen(char(dest + "/README.md"), 'w');
fwrite(fid, char(strjoin(text, newline)));
fclose(fid);
end

function build_and_run(dest, pde, numpde, prefix)
% Configure, build, and run the bundle against the local Exasim install to
% verify it works before hand-off.
%
% The build and the run output go to a throwaway scratch directory so the
% shipped bundle stays pristine (source + datain + empty dataout); the
% scratch only proves the relocatable CMakeLists builds and the solver runs.
disp("Verify exported app: build + run against the local Exasim install...");
cmakecmd = resolve_cmake(prefix);
work = tempname();
mkdir(char(work));
cleanup = onCleanup(@() rmdir_quiet(work));

bdir = string(work) + "/build";
comstr = cmakecmd + " -S " + dest + " -B " + bdir + ...
         " -DExasim_DIR=" + prefix + "/lib/cmake/Exasim";
runchecked(comstr);

jobs = getenv('JOBS');
if isempty(jobs), jobs = num2str(feature('numcores')); end
runchecked(cmakecmd + " --build " + bdir + " --parallel " + jobs);

exe = bdir + "/exasimapp";
if ~exist(char(exe), 'file')
    error("Bundle build did not produce %s.", exe);
end

datain = dest + "/datain/";
dataout = string(work) + "/out";
if pde.mpiprocs > 1
    mpirun = "mpirun";
    if isfield(pde, 'mpirun') && ~isempty(pde.mpirun)
        mpirun = string(pde.mpirun);
    end
    mpitxt = char(bdir + "/mpiexec.txt");
    if exist(mpitxt, 'file')
        discovered = strip(string(fileread(mpitxt)));
        if strlength(discovered) > 0, mpirun = discovered; end
    end
    runstr = mpirun + " -np " + string(pde.mpiprocs) + " " + exe + ...
             " " + string(numpde) + " " + datain + " " + dataout;
else
    runstr = exe + " " + string(numpde) + " " + datain + " " + dataout;
end

eval("!" + runstr);

% cdir = pwd();
% cd(char(work));
% [status, output] = system(char(runstr));
% disp(output);
% cd(char(cdir));
% if status ~= 0
%     error("Bundle verification run failed (exit %d): %s", status, runstr);
% end

produced = dir(char(string(work) + "/out*"));
if isempty(produced)
    error("Bundle verification run produced no output.");
end
disp("Exported app verified (built and ran against the local install).");
end

% ---------------------------------------------------------------------------

function copytree(src, dst)
% Mirror src into dst (replacing dst).
if exist(char(dst), 'dir'), rmdir(char(dst), 's'); end
mkdir(char(dst));
copyfile(char(src), char(dst));
end

function rmdir_quiet(p)
if exist(char(p), 'dir')
    try, rmdir(char(p), 's'); catch, end
end
end

function runchecked(cmd)
[status, output] = system(char(cmd));
disp(output);
if status ~= 0
    error("Command failed (exit %d): %s", status, cmd);
end
end

function p = absolutepath_str(p)
% Absolute path of p (which may not yet exist).
p = char(p);
if ~(ispc && numel(p) >= 2 && p(2) == ':') && ~(~ispc && ~isempty(p) && p(1) == '/')
    p = fullfile(pwd(), p);
end
end

function rendertemplate(srcfile, dstfile, subs)
text = fileread(char(srcfile));
for i = 1:size(subs, 1)
    text = strrep(text, char("@" + subs{i,1} + "@"), char(string(subs{i,2})));
end
% Write only on change so unchanged apps don't dirty mtimes (recompiles).
if exist(char(dstfile), 'file') == 2 && strcmp(fileread(char(dstfile)), text)
    return;
end
fid = fopen(char(dstfile), 'w');
fwrite(fid, text);
fclose(fid);
end

function cmakecmd = resolve_cmake(prefix)
% Mirror cmakecompile.m: prefer the cmake recorded at install time, then
% discover one on this machine, falling back to a bare "cmake".
resolved = "";
cmakerec = char(prefix + "/lib/cmake/Exasim/cmake_command.txt");
if exist(cmakerec, 'file') == 2
    rec = strtrim(string(fileread(cmakerec)));
    if strlength(rec) > 0 && exist(char(rec), 'file') == 2
        resolved = rec;
    end
end
if strlength(resolved) == 0
    [st, out] = system('command -v cmake');
    if st == 0 && strlength(strtrim(string(out))) > 0
        resolved = strtrim(string(out));
    else
        for cand = ["/opt/homebrew/bin/cmake", "/usr/local/bin/cmake", ...
                    "/opt/local/bin/cmake", "/usr/bin/cmake"]
            if exist(char(cand), 'file') == 2
                resolved = cand; break;
            end
        end
    end
end
if strlength(resolved) > 0
    cmakecmd = """" + resolved + """";
else
    cmakecmd = "cmake";
end
end
