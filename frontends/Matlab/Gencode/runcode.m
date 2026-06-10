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

exe = pde.builddir + "/build/exasimapp";
if ~exist(char(exe), 'file')
    error("Solver executable not found at %s; run cmakecompile(pde) first.", exe);
end

DataPath = pde.datapath;
mystr = " " + num2str(numpde) + " ";
if numpde>100 % two-domain problems
  numpde = 2;
end
if numpde==1
    mystr = mystr + DataPath + "/datain/ " + DataPath + "/dataout/out";
else
    for i = 1:numpde
        mystr = mystr + DataPath + "/datain" + num2str(i) + "/ " + DataPath + "/dataout" + num2str(i) + "/out";
        mystr = mystr + " ";
    end
end

if mpiprocs==1
    runstr = exe + mystr;
else
    runstr = pde.mpirun + " -np " + string(mpiprocs) + " " + exe + mystr;
end

cdir = pwd();
cd(char(DataPath));
tic
[status, output] = system(char(runstr));
disp(output);
toc
cd(char(cdir));

if status ~= 0
    error("Solver failed (exit %d): %s", status, runstr);
end

end
