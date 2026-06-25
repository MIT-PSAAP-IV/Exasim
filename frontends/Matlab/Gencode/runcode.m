function runstr = runcode(pde, numpde, mpiprocs)
% Run the solver executable built by cmakecompile.
%
% Uses the legacy CLI: exasimapp <numpde> <datain>/ <dataout>/out, with
% datain/dataout under pde.datapath and the executable in the hidden
% pde.builddir. Errors on a nonzero solver exit code.

if nargin<3
  mpiprocs = pde.mpiprocs;
end

disp("Run C++ Exasim code ...")

if isfield(pde, 'combinedmodel') && pde.combinedmodel
    exe = model_builddir(pde) + "/build/exasimapp";   % external-model path
else
    exe = string(pde.builddir) + "/build/exasimapp";  % legacy path
end
if ~exist(char(exe), 'file')
    error("Solver executable not found at %s; run cmakecompile(pde) first.", exe);
end

executionmode = 0;
if isfield(pde, 'executionmode')
    executionmode = pde.executionmode;
end
if executionmode == 0
    modestring = " ";
elseif executionmode == 1
    modestring = " postprocess";
else
    error("Unsupported executionmode=%d. Use 0 for solve or 1 for postprocess.", executionmode);
end

DataPath = pde.datapath;
mystr = modestring + " " + num2str(numpde) + " ";
if numpde>100 % two-domain problems
  numpde = 2;
end
if numpde==1
    % per-model datain/dataout (sibling strn dirs; "" for model 0 -> datain/)
    strn = model_strn(pde);
    if isfield(pde, 'dataoutpath') && strlength(string(pde.dataoutpath)) > 0
        dataout = string(pde.dataoutpath);
    else
        dataout = DataPath + "/dataout" + strn;
    end
    mystr = mystr + DataPath + "/datain" + strn + "/ " + dataout + "/out";
else
    for i = 1:numpde
        mystr = mystr + DataPath + "/datain" + num2str(i) + "/ " + DataPath + "/dataout" + num2str(i) + "/out";
        mystr = mystr + " ";
    end
end

if mpiprocs==1
    runstr = exe + mystr;
else
    % Prefer the MPI launcher CMake discovered at build-configure time
    % (portable); fall back to the frontend-detected pde.mpirun.
    mpirun = pde.mpirun;
    mpitxt = char(pde.builddir + "/build/mpiexec.txt");
    if exist(mpitxt, 'file')
        discovered = strip(string(fileread(mpitxt)));
        if strlength(discovered) > 0
            mpirun = discovered;
        end
    end
    runstr = mpirun + " -np " + string(mpiprocs) + " " + exe + mystr;
end

eval("!" + runstr);

% cdir = pwd();
% cd(char(DataPath));
% tic
% [status, output] = system(char(runstr));
% disp(output);
% toc
% cd(char(cdir));
% 
% if status ~= 0
%     error("Solver failed (exit %d): %s", status, runstr);
% end

end
