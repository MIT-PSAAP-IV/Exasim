function comstr = cmakecompile(pde,mpiprocs)

if nargin<2
  mpiprocs = pde.mpiprocs;
end

disp("Compile C++ Exasim code using cmake...")

cdir = pwd();
buildpath = char(pde.buildpath);
if exist(buildpath, "dir") == 0
    mkdir(buildpath);
end
cd(buildpath);

sourcepath = char(pde.exasimpath + "/examples");

if exist(fullfile(buildpath, "exasimfe"), "file")
    delete(fullfile(buildpath, "exasimfe"));
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

% Use system() and check the exit code so configure / build failures
% surface immediately rather than hiding behind missing-binary errors
% downstream. The previous `eval("!cmake ...")` swallowed nonzero exit.
status = system(char(comstr));
if status ~= 0
    cd(char(cdir));
    error("Exasim:cmakecompile", "cmake configure failed (exit %d)", status);
end
status = system("cmake --build . --target exasimfe --verbose");
if status ~= 0
    cd(char(cdir));
    error("Exasim:cmakecompile", "cmake --build exasimfe failed (exit %d)", status);
end

cd(char(cdir));
