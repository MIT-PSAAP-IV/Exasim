function comstr = cmakecompile(pde, ~)
% Build the solver executable for the generated model.
%
% Renders the installed frontend-app templates into the hidden pde.builddir
% and builds them against the installed Exasim package via
% find_package(Exasim). The kkgencode step must have written the kernel set
% to <builddir>/kernels first. Returns the configure command.

disp("Compile C++ Exasim code using cmake...")

cdir = pwd();

if pde.sharedbuild==0  
  exasimroot = exasim_path();
  sourcepath = fullfile(exasimroot, "examples", "exasimfe");
  targetpath = fullfile(cdir, "exasim");
  if exist(targetpath, "dir") == 0
      mkdir(targetpath);
  end

  files = ["exasimfeapp.cpp", "CMakeLists.txt", "frontendprovider.cpp"];
  for i = 1:numel(files)
      copyfile(fullfile(sourcepath, files(i)), fullfile(targetpath, files(i)));
  end
end

mpiprocs = pde.mpiprocs;
sourcepath = pde.builddir;

buildpath = char(pde.builddir + "/build");
if exist(buildpath, "dir") == 0
    mkdir(buildpath);
end
cd(buildpath);

if exist(fullfile(buildpath, "exasimapp"), "file")
    delete(fullfile(buildpath, "exasimapp"));
end

if mpiprocs==1
  if pde.platform == "gpu"
    comstr = "cmake -S " + sourcepath + " -B . -D EXASIM_MPI=OFF -D EXASIM_CUDA=ON";
  elseif pde.platform == "hip"
    comstr = "cmake -S " + sourcepath + " -B . -D CMAKE_CXX_COMPILER=hipcc -D EXASIM_MPI=OFF -D EXASIM_HIP=ON";
  else
    comstr = "cmake -S " + sourcepath + " -B . -D EXASIM_MPI=OFF";
  end
else
  if pde.platform == "gpu"
    comstr = "cmake -S " + sourcepath + " -B . -D EXASIM_MPI=ON -D EXASIM_CUDA=ON";
  elseif pde.platform == "hip"
    comstr = "cmake -S " + sourcepath + " -B . -D CMAKE_CXX_COMPILER=hipcc -D EXASIM_MPI=ON -D EXASIM_HIP=ON";
  else
    comstr = "cmake -S " + sourcepath + " -B . -D EXASIM_MPI=ON";
  end
end

prefix = exasim_install_prefix();
comstr = comstr + " -DExasim_DIR=" + prefix;

% Use system() and check the exit code so configure / build failures
% surface immediately rather than hiding behind missing-binary errors
% downstream. The previous `eval("!cmake ...")` swallowed nonzero exit.
status = system(char(comstr));
if status ~= 0
    cd(char(cdir));
    error("Exasim:cmakecompile", "cmake configure failed (exit %d)", status);
end
status = system("cmake --build . --target exasimapp --verbose");
if status ~= 0
    cd(char(cdir));
    error("Exasim:cmakecompile", "cmake --build exasimapp failed (exit %d)", status);
end

cd(char(cdir));

end

% disp("Compile C++ Exasim code against the installed Exasim package...");
% 
% if pde.sharedbuild==1
%   comstr = cmakecompilesharedbuild(pde);
%   return;
% end
% 
% prefix = exasim_install_prefix();
% pde.exasimpath = prefix;
% 
% builddir = pde.builddir;
% kernels = builddir + "/kernels";
% if ~exist(char(kernels), 'dir')
%     error("No generated kernels at %s; run kkgencode(pde) first.", kernels);
% end
% 
% if pde.platform == "gpu"
%     if pde.mpiprocs > 1, variant = "gpumpi"; else, variant = "gpu"; end
% else
%     if pde.mpiprocs > 1, variant = "cpumpi"; else, variant = "cpu"; end
% end
% 
% tmpl = prefix + "/lib/cmake/Exasim/frontend-app";
% subs = {"EXASIM_VARIANT", variant; "MODEL_ID", string(pde.modelid); "KERNEL_DIR", kernels};
% rendertemplate(tmpl + "/CMakeLists.txt.in", builddir + "/CMakeLists.txt", subs);
% rendertemplate(tmpl + "/main.cpp.in", builddir + "/main.cpp", subs);
% 
% bdir = builddir + "/build";
% exe = bdir + "/exasimapp";
% comstr = "cmake -S " + builddir + " -B " + bdir + ...
%          " -DExasim_DIR=" + prefix + "/lib/cmake/Exasim";
% 
% % Hash the model inputs (kernels + rendered app sources + the install);
% % if nothing changed since the last successful build, skip cmake entirely.
% stamp = char(bdir + "/.exasim_model_hash");
% % Templates rather than rendered files: rendered sources embed absolute
% % paths, which would make the digest directory-specific.
% digest = modelhash(kernels, {char(tmpl + "/CMakeLists.txt.in"), char(tmpl + "/main.cpp.in")}, prefix, variant, pde.modelid);
% if exist(char(exe), 'file') == 2 && exist(stamp, 'file') == 2 ...
%         && strcmp(strtrim(fileread(stamp)), digest)
%     disp("Model unchanged (hash match); skipping build.");
%     return;
% end
% 
% % Per-user model cache: the relocatable (libfrontend_model, exasimapp) pair
% % built for this modelid+digest by any earlier app run.
% cachedir = fullfile(cacheroot(), num2str(pde.modelid), digest);
% cached = cachefiles(cachedir);
% if ~isempty(cached)
%     if ~exist(char(bdir), 'dir'), mkdir(char(bdir)); end
%     for i = 1:numel(cached)
%         copyfile(cached{i}, char(bdir));
%     end
%     fid = fopen(stamp, 'w'); fwrite(fid, digest); fclose(fid);
%     disp("Model cache hit (" + string(cachedir) + "); skipping build.");
%     return;
% end
% 
% runchecked(comstr);
% 
% jobs = getenv('JOBS');
% if isempty(jobs), jobs = num2str(feature('numcores')); end
% runchecked("cmake --build " + bdir + " --parallel " + jobs);
% 
% if ~exist(char(exe), 'file')
%     error("Build did not produce %s.", exe);
% end
% fid = fopen(stamp, 'w');
% fwrite(fid, digest);
% fclose(fid);
% 
% % Populate the cache for other app directories / future runs.
% libs = dir(char(bdir + "/libfrontend_model.*"));
% if ~isempty(libs)
%     if ~exist(char(cachedir), 'dir'), mkdir(char(cachedir)); end
%     for i = 1:numel(libs)
%         copyfile(fullfile(libs(i).folder, libs(i).name), char(cachedir));
%     end
%     copyfile(char(exe), char(cachedir));
% end
% end
% 
% % Per-user cache of built model libraries (EXASIM_CACHE_DIR overrides).
% function root = cacheroot()
% env = getenv('EXASIM_CACHE_DIR');
% if isempty(env)
%     root = fullfile(char(java.lang.System.getProperty('user.home')), '.exasim', 'cache');
% else
%     root = fullfile(env, 'cache');
% end
% end
% 
% % Return the cached (lib, exe) pair as a cellstr, or {} if incomplete.
% function files = cachefiles(cachedir)
% files = {};
% if ~exist(char(cachedir), 'dir'), return; end
% libs = dir(char(string(cachedir) + "/libfrontend_model.*"));
% exe = fullfile(char(cachedir), 'exasimapp');
% if isempty(libs) || exist(exe, 'file') ~= 2, return; end
% files = cell(1, numel(libs) + 1);
% for i = 1:numel(libs)
%     files{i} = fullfile(libs(i).folder, libs(i).name);
% end
% files{end} = exe;
% end
% 
% % SHA-256 over the kernel set, the rendered app sources, and the identity
% % of the Exasim install they build against.
% function h = modelhash(kernelsdir, extra_files, prefix, variant, modelid)
% md = java.security.MessageDigest.getInstance('SHA-256');
% md.update(uint8([char(string(variant)) '|' char(string(modelid))]));
% files = dir(char(kernelsdir));
% [~, order] = sort({files.name});
% files = files(order);
% for i = 1:numel(files)
%     if files(i).isdir, continue; end
%     md.update(uint8(files(i).name));
%     fid = fopen(fullfile(files(i).folder, files(i).name), 'r');
%     md.update(fread(fid, inf, 'uint8=>uint8'));
%     fclose(fid);
% end
% for i = 1:numel(extra_files)
%     fid = fopen(extra_files{i}, 'r');
%     md.update(fread(fid, inf, 'uint8=>uint8'));
%     fclose(fid);
% end
% md.update(uint8(char(prefix)));
% targets = fullfile(char(prefix), 'lib', 'cmake', 'Exasim', 'ExasimTargets.cmake');
% if exist(targets, 'file') == 2
%     s = dir(targets);
%     md.update(uint8(sprintf('%d:%d', round(s.datenum*86400), s.bytes)));
% end
% h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []));
% end
% 
% function rendertemplate(srcfile, dstfile, subs)
% text = fileread(char(srcfile));
% for i = 1:size(subs, 1)
%     text = strrep(text, char("@" + subs{i,1} + "@"), char(string(subs{i,2})));
% end
% % Write only on change so unchanged apps don't dirty mtimes (recompiles).
% if exist(char(dstfile), 'file') == 2 && strcmp(fileread(char(dstfile)), text)
%     return;
% end
% fid = fopen(char(dstfile), 'w');
% fwrite(fid, text);
% fclose(fid);
% end
% 
% function runchecked(cmd)
% [status, output] = system(char(cmd));
% disp(output);
% if status ~= 0
%     error("Command failed (exit %d): %s", status, cmd);
% end
% end
