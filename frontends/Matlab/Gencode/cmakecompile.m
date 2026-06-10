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
comstr = "cmake -S " + builddir + " -B " + bdir + ...
         " -DExasim_DIR=" + prefix + "/lib/cmake/Exasim";

% Hash the model inputs; if nothing changed since the last successful
% build, skip cmake entirely.
stamp = char(bdir + "/.exasim_model_hash");
digest = modelhash(kernels, {char(builddir + "/CMakeLists.txt"), char(builddir + "/main.cpp")});
if exist(char(exe), 'file') == 2 && exist(stamp, 'file') == 2 ...
        && strcmp(strtrim(fileread(stamp)), digest)
    disp("Model unchanged (hash match); skipping build.");
    return;
end

runchecked(comstr);

jobs = getenv('JOBS');
if isempty(jobs), jobs = num2str(feature('numcores')); end
runchecked("cmake --build " + bdir + " --parallel " + jobs);

if ~exist(char(exe), 'file')
    error("Build did not produce %s.", exe);
end
fid = fopen(stamp, 'w');
fwrite(fid, digest);
fclose(fid);
end

% SHA-256 over the kernel set and the rendered app sources.
function h = modelhash(kernelsdir, extra_files)
md = java.security.MessageDigest.getInstance('SHA-256');
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
