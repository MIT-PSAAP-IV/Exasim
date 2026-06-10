function runcode(pde, numpde=1)

display("Run C++ Exasim code ...")

cdir = pwd(); 
cd(pde.buildpath);

buildpath = pde.buildpath;
DataPath = buildpath;
mpirun = pde.mpirun;
pdenum = string(numpde) * " ";

if pde.mpiprocs==1
    str = "./exasimfe " * pdenum * DataPath * "/datain/ " * DataPath * "/dataout/out";
else
    str = mpirun * " -np " * string(pde.mpiprocs) * " ./exasimfe " * pdenum * DataPath * "/datain/ " * DataPath * "/dataout/out";
end

run(string2cmd(str));

cd(cdir);

return str;

end
