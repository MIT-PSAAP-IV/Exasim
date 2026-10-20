function runstr = runcode_coupled(pdes)
% Run a two-model interface solve through the combined executable.
% The split marker 100+n0 tells the backend that world ranks [0,n0) own
% model 0 and the remaining ranks own model 1.

if numel(pdes) ~= 2
    error("Coupled interface solves support exactly two models.");
end

p0 = pdes{1};
exe = string(p0.builddir) + "/combined/build/exasimapp";
if ~exist(char(exe), 'file')
    error("Combined executable not found at %s; run cmakecompile_combined(pdes) first.", exe);
end

mystr = " " + string(100 + pdes{1}.mpiprocs) + " ";
for m = 1:2
    strn = model_strn(pdes{m});
    mystr = mystr + string(p0.datapath) + "/datain" + strn + "/ " + ...
            string(p0.datapath) + "/dataout" + strn + "/out ";
end

mpirun = p0.mpirun;
mpitxt = char(string(p0.builddir) + "/combined/build/mpiexec.txt");
if exist(mpitxt, 'file')
    discovered = strip(string(fileread(mpitxt)));
    if strlength(discovered) > 0
        mpirun = discovered;
    end
end

mpiprocs = pdes{1}.mpiprocs + pdes{2}.mpiprocs;
runstr = mpirun + " -np " + string(mpiprocs) + " " + exe + mystr;
eval("!" + runstr);
end
