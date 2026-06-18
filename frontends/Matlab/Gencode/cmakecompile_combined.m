function comstr = cmakecompile_combined(pdes)
% Build ONE solver executable that runs several generated models together
% (combined multi-PDE). Each pde in the cell array `pdes` must already have its
% kernels generated (kkgencode) into its own per-model build dir and a distinct
% modelnumber (hence distinct modelid). Renders the frontend-app-combined
% templates into <builddir>/combined and builds against the installed Exasim.
% Mirrors the Python frontend's cmakecompile_combined.

disp("Compile combined multi-PDE Exasim app against the installed Exasim package...");

prefix = exasim_install_prefix();
cmakecmd = exasim_cmake_command(prefix);

p0 = pdes{1};
builddir = string(p0.builddir) + "/combined";
if ~exist(char(builddir), 'dir'), mkdir(char(builddir)); end

n = numel(pdes);
ids = strings(1, n);
kdirs = strings(1, n);
for m = 1:n
    pdes{m}.exasimpath = prefix;
    ids(m) = string(resolve_modelid(pdes{m}));
    kd = model_builddir(pdes{m}) + "/kernels";
    if ~exist(char(kd), 'dir')
        error("No generated kernels at %s; run kkgencode(pde) first.", kd);
    end
    kdirs(m) = """" + kd + """";
end
if numel(unique(ids)) ~= n
    error("combined models need distinct modelids (one per slot); got %s", strjoin(ids, " "));
end

if p0.platform == "gpu"
    if p0.mpiprocs > 1, variant = "gpumpi"; else, variant = "gpu"; end
else
    if p0.mpiprocs > 1, variant = "cpumpi"; else, variant = "cpu"; end
end

tmpl = prefix + "/lib/cmake/Exasim/frontend-app-combined";
subs = {"EXASIM_VARIANT", variant; ...
        "MODEL_IDS",     strjoin(ids, ", "); ...    % main.cpp: {100, 101}
        "MODEL_ID_LIST", strjoin(ids, " "); ...     % CMake: IDS 100 101
        "KERNEL_DIRS",   strjoin(kdirs, " ")};      % CMake: KERNELS_DIRS
rendertemplate(tmpl + "/CMakeLists.txt.in", builddir + "/CMakeLists.txt", subs);
rendertemplate(tmpl + "/main.cpp.in", builddir + "/main.cpp", subs);

bdir = builddir + "/build";
exe = bdir + "/exasimapp";
comstr = cmakecmd + " -S " + builddir + " -B " + bdir + ...
         " -DExasim_DIR=" + prefix + "/lib/cmake/Exasim";
runchecked(comstr);

jobs = getenv('JOBS');
if isempty(jobs), jobs = num2str(feature('numcores')); end
runchecked(cmakecmd + " --build " + bdir + " --parallel " + jobs);

if ~exist(char(exe), 'file')
    error("Build did not produce %s.", exe);
end
end

function rendertemplate(srcfile, dstfile, subs)
text = fileread(char(srcfile));
for i = 1:size(subs, 1)
    text = strrep(text, char("@" + subs{i,1} + "@"), char(string(subs{i,2})));
end
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
