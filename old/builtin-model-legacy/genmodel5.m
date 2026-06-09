% Add Exasim to Matlab search path
cdir = pwd(); ii = strfind(cdir, "Exasim");
run(cdir(1:(ii+5)) + "/install/setpath.m");

pde = initializeexasim();
pde.builtinmodelID = 5;
pde.modelfile = "pdemodel" + num2str(pde.builtinmodelID);  
pde.hybrid = 1;
pde.nd = 2;
pde.ncu = 4;
pde.ncq = 8;
pde.ncw = 0;
pde.ncv = 0;
pde.ntau = 1;
pde.nmu = 6;
pde.neta = 0;

kkgenmodel(pde);

