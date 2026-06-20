function runstr = runcode_combined(pdes, mpiprocs)
% Run the combined multi-PDE executable built by cmakecompile_combined.
% Emits the per-model CLI the solver's ParseInputs expects:
%   exasimapp N datain<s0>/ dataout<s0>/out datain<s1>/ dataout<s1>/out ...
% one (datain, dataout) pair per model slot (sibling strn dirs; "" for slot 0).
% Mirrors the Python frontend's runcode_combined.

p0 = pdes{1};
if nargin < 2, mpiprocs = p0.mpiprocs; end

n = numel(pdes);
exe = string(p0.builddir) + "/combined/build/exasimapp";
if ~exist(char(exe), 'file')
    error("Combined executable not found at %s; run cmakecompile_combined(pdes) first.", exe);
end

DataPath = p0.datapath;
mystr = " " + num2str(n) + " ";
for m = 1:n
    strn = model_strn(pdes{m});
    mystr = mystr + DataPath + "/datain" + strn + "/ " + DataPath + "/dataout" + strn + "/out ";
end

if mpiprocs == 1
    runstr = exe + mystr;
else
    mpirun = p0.mpirun;
    mpitxt = char(string(p0.builddir) + "/combined/build/mpiexec.txt");
    if exist(mpitxt, 'file')
        discovered = strip(string(fileread(mpitxt)));
        if strlength(discovered) > 0
            mpirun = discovered;
        end
    end
    runstr = mpirun + " -np " + string(mpiprocs) + " " + exe + mystr;
end

eval("!" + runstr);
end
