% Add Exasim to Matlab search path
cdir = pwd(); ii = strfind(cdir, "Exasim");
run(cdir(1:(ii+5)) + "/install/setpath.m");

pde = initializeexasim();
pde.modelfile = "pdemodel_cart";  
pde.hybrid = 1;
pde.nd = 1;
pde.ncu = 7;
pde.ncq = pde.ncu*pde.nd;
pde.ncw = 1;
pde.ncv = 1;
pde.ntau = 1;
pde.nmu = 12;
pde.neta = pde.ncu + 10;
kkgenmodel(pde, "cart1d");

pde.nd = 2;
pde.ncu = 5+pde.nd+1;
pde.ncq = pde.ncu*pde.nd;
pde.neta = pde.ncu + 10;
kkgenmodel(pde, "cart2d");

pde.nd = 3;
pde.ncu = 5+pde.nd+1;
pde.ncq = pde.ncu*pde.nd;
pde.neta = pde.ncu  + 10;
kkgenmodel(pde, "cart3d");


pde.modelfile = "pdemodel_axial";  
pde.hybrid = 1;
pde.nd = 1;
pde.ncu = 7;
pde.ncq = pde.ncu*pde.nd;
pde.ncw = 1;
pde.ncv = 1;
pde.ntau = 1;
pde.nmu = 12;
pde.neta = pde.ncu  + 10;
kkgenmodel(pde, "axial1d");

pde.nd = 2;
pde.ncu = 5+pde.nd+1;
pde.ncq = pde.ncu*pde.nd;
pde.neta = pde.ncu  + 10;
kkgenmodel(pde, "axial2d");

pde.nd = 3;
pde.ncu = 5+pde.nd+1;
pde.ncq = pde.ncu*pde.nd;
pde.neta = pde.ncu  + 10;
kkgenmodel(pde, "axial3d");


