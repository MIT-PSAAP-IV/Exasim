function comstr = cmakecompile(pde, ~)
% Build the solver executable for the generated model.
%
% Renders the installed frontend-app templates into the hidden pde.builddir
% and builds them against the installed Exasim package via
% find_package(Exasim). The kkgencode step must have written the kernel set
% to <builddir>/kernels first. Returns the configure command.

disp("Compile C++ Exasim code against the installed Exasim package...");

prefix = exasim_install_prefix();
pde.exasimpath = prefix;

% Use the absolute path of the cmake that built/installed Exasim (recorded at
% install time) so this works when MATLAB is launched from the GUI app, whose
% PATH does not include the Homebrew/conda cmake. If that record is missing or
% its path no longer exists (a relocated install — recorded cmake belongs to
% another machine), discover a cmake on this machine. Last resort: bare "cmake".
resolved = "";
cmakerec = char(prefix + "/lib/cmake/Exasim/cmake_command.txt");
if exist(cmakerec, 'file') == 2
    rec = strtrim(string(fileread(cmakerec)));
    if strlength(rec) > 0 && exist(char(rec), 'file') == 2
        resolved = rec;
    end
end
if strlength(resolved) == 0
    [st, out] = system('command -v cmake');   % PATH (incl. exasim_setup additions)
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
    cmakecmd = """" + resolved + """";   % quote in case the path has spaces
else
    cmakecmd = "cmake";
end

builddir = pde.builddir;
kernels = builddir + "/kernels";
if ~exist(char(kernels), 'dir')
    error("No generated kernels at %s; run kkgencode(pde) first.", kernels);
end

if pde.platform == "gpu"
    if pde.mpiprocs > 1, variant = "gpumpi"; else, variant = "gpu"; end
else
    if pde.mpiprocs > 1, variant = "cpumpi"; else, variant = "cpu"; end
end

tmpl = prefix + "/lib/cmake/Exasim/frontend-app";
subs = {"EXASIM_VARIANT", variant; "MODEL_ID", string(pde.modelid); "KERNEL_DIR", kernels};
rendertemplate(tmpl + "/CMakeLists.txt.in", builddir + "/CMakeLists.txt", subs);
rendertemplate(tmpl + "/main.cpp.in", builddir + "/main.cpp", subs);

bdir = builddir + "/build";
exe = bdir + "/exasimapp";
comstr = cmakecmd + " -S " + builddir + " -B " + bdir + ...
         " -DExasim_DIR=" + prefix + "/lib/cmake/Exasim";

% Hash the model inputs (kernels + rendered app sources + the install);
% if nothing changed since the last successful build, skip cmake entirely.
stamp = char(bdir + "/.exasim_model_hash");
% Templates rather than rendered files: rendered sources embed absolute
% paths, which would make the digest directory-specific.
digest = modelhash(kernels, {char(tmpl + "/CMakeLists.txt.in"), char(tmpl + "/main.cpp.in")}, prefix, variant, pde.modelid);
if exist(char(exe), 'file') == 2 && exist(stamp, 'file') == 2 ...
        && strcmp(strtrim(fileread(stamp)), digest)
    disp("Model unchanged (hash match); skipping build.");
    return;
end

% Per-user model cache: the relocatable (libfrontend_model, exasimapp) pair
% built for this modelid+digest by any earlier app run.
cachedir = fullfile(cacheroot(), num2str(pde.modelid), digest);
cached = cachefiles(cachedir);
if ~isempty(cached)
    if ~exist(char(bdir), 'dir'), mkdir(char(bdir)); end
    for i = 1:numel(cached)
        copyfile(cached{i}, char(bdir));
    end
    fid = fopen(stamp, 'w'); fwrite(fid, digest); fclose(fid);
    disp("Model cache hit (" + string(cachedir) + "); skipping build.");
    return;
end

runchecked(comstr);

jobs = getenv('JOBS');
if isempty(jobs), jobs = num2str(feature('numcores')); end
runchecked(cmakecmd + " --build " + bdir + " --parallel " + jobs);

if ~exist(char(exe), 'file')
    error("Build did not produce %s.", exe);
end
fid = fopen(stamp, 'w');
fwrite(fid, digest);
fclose(fid);

% Populate the cache for other app directories / future runs.
libs = dir(char(bdir + "/libfrontend_model.*"));
if ~isempty(libs)
    if ~exist(char(cachedir), 'dir'), mkdir(char(cachedir)); end
    for i = 1:numel(libs)
        copyfile(fullfile(libs(i).folder, libs(i).name), char(cachedir));
    end
    copyfile(char(exe), char(cachedir));
end
end

% Per-user cache of built model libraries (EXASIM_CACHE_DIR overrides).
function root = cacheroot()
env = getenv('EXASIM_CACHE_DIR');
if isempty(env)
    root = fullfile(char(java.lang.System.getProperty('user.home')), '.exasim', 'cache');
else
    root = fullfile(env, 'cache');
end
end

% Return the cached (lib, exe) pair as a cellstr, or {} if incomplete.
function files = cachefiles(cachedir)
files = {};
if ~exist(char(cachedir), 'dir'), return; end
libs = dir(char(string(cachedir) + "/libfrontend_model.*"));
exe = fullfile(char(cachedir), 'exasimapp');
if isempty(libs) || exist(exe, 'file') ~= 2, return; end
files = cell(1, numel(libs) + 1);
for i = 1:numel(libs)
    files{i} = fullfile(libs(i).folder, libs(i).name);
end
files{end} = exe;
end

% SHA-256 over the kernel set, the rendered app sources, and the identity
% of the Exasim install they build against.
function h = modelhash(kernelsdir, extra_files, prefix, variant, modelid)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(uint8([char(string(variant)) '|' char(string(modelid))]));
files = dir(char(kernelsdir));
[~, order] = sort({files.name});
files = files(order);
for i = 1:numel(files)
    if files(i).isdir, continue; end
    md.update(uint8(files(i).name));
    fid = fopen(fullfile(files(i).folder, files(i).name), 'r');
    md.update(fread(fid, inf, 'uint8=>uint8'));
    fclose(fid);
end
for i = 1:numel(extra_files)
    fid = fopen(extra_files{i}, 'r');
    md.update(fread(fid, inf, 'uint8=>uint8'));
    fclose(fid);
end
md.update(uint8(char(prefix)));
targets = fullfile(char(prefix), 'lib', 'cmake', 'Exasim', 'ExasimTargets.cmake');
if exist(targets, 'file') == 2
    s = dir(targets);
    md.update(uint8(sprintf('%d:%d', round(s.datenum*86400), s.bytes)));
end
h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []));
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

function runchecked(cmd)
[status, output] = system(char(cmd));
disp(output);
if status ~= 0
    error("Command failed (exit %d): %s", status, cmd);
end
end
