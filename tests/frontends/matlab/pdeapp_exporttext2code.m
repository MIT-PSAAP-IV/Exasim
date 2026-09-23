% Text2Code export fixture (MATLAB): mirrors tests/frontends/python/pdeapp_exporttext2code.py.
% Exports a small Poisson application as a Text2Code package through the MATLAB
% exporter (exporttext2code -> writeinputfile). The harness checks the package
% contents -- including that pdeapp.txt carries uniformrefinementlevel -- and
% regenerates datain from it with the installed text2code.

[pde,mesh] = initializeexasim();

pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.mpiprocs = 1;
pde.hybrid = 1;
pde.porder = 1;
pde.pgauss = 2;
pde.physicsparam = 1;
pde.physicsparamsweep = [1; 2];
pde.tau = 1.0;
% Exported to pdeapp.txt and applied by text2code (grid + xdg/udg/vdg/wdg are refined there).
pde.uniformrefinementlevel = 1;
pde.AV = 1;
pde.AVdistfunction = 1;
pde.distanceboundaryconditions = [1];
pde.AVsmoothingMethod = 1;
pde.AVHelmholtzCoeff = 0.375;
pde.avparam1 = [9.0, 8.0];
pde.avparam2 = [7.0, 6.0];
pde.AVcontinuationIter = 5;
pde.AVcontinuationLogScale = 1.0;
pde.AVcoeffStart = 0.06;
pde.AVcoeffEnd = 0.015;

[mesh.p,mesh.t] = squaremesh(2,2,1,1);
mesh.boundaryexpr = {@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-1)<1e-8, @(p) abs(p(2,:)-1)<1e-8, @(p) abs(p(1,:))<1e-8};
mesh.boundarycondition = [1;1;1;1];

% Include external and auxiliary fields so the package also carries vdg.bin and wdg.bin.
mesh.dgnodes = createdgnodes(mesh.p, mesh.t, zeros(4, size(mesh.t,2)), [], [], pde.porder);
npe = size(mesh.dgnodes, 1);
ne = size(mesh.dgnodes, 3);
mesh.udg = zeros(npe, pde.ncu, ne);
mesh.vdg = ones(npe, 1, ne);
mesh.wdg = 2*ones(npe, 1, ne);

% Compare the exported package against ordinary MATLAB preprocessing.
pde_native = pde;
pde_native.datapath = string(fullfile(pwd, "native_preprocessing"));
preprocessing(pde_native, mesh);

dest = fullfile(pwd, "text2code_package");
exporttext2code(pde, mesh, dest);
fprintf("TEXT2CODE EXPORT PACKAGE: %s\n", dest);
