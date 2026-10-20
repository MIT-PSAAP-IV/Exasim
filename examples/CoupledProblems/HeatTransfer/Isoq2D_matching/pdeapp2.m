% Configure the axisymmetric solid heat equation on a Gmsh triangle mesh.
if ~exist('fluidMesh','var') || ~exist('fluidPde','var')
    error('pdeapp2 requires fluidMesh and fluidPde from pdeapp1.');
end

[pde2,~] = initializeexasim();
pde2.model = "ModelD";
pde2.modelfile = "pdemodel2";
pde2.platform = fluidPde.platform;
pde2.mpiprocs = 2;
pde2.hybrid = 1;
pde2.debugmode = fluidPde.debugmode;
pde2.nd = 2;
pde2.gencode = 1;
pde2.Cxxpreprocessing = 0;

pde2.porder = fluidPde.porder;
pde2.pgauss = fluidPde.pgauss;
kappa = 0.01;
solidTemperature = Twall/Tref*Tinf;
pde2.physicsparam = [kappa solidTemperature];
pde2.tau = 1.0;
pde2.GMRESrestart = fluidPde.GMRESrestart;
pde2.GMRESortho = fluidPde.GMRESortho;
pde2.linearsolvertol = fluidPde.linearsolvertol;
pde2.linearsolveriter = fluidPde.linearsolveriter;
pde2.preconditioner = fluidPde.preconditioner;
pde2.RBdim = fluidPde.RBdim;
pde2.ppdegree = fluidPde.ppdegree;
% The monolithic fluid-solid solve has one Krylov/Newton history. Keep both
% models' solver controls identical so every coupled-world collective occurs
% in the same order and the shared solve has unambiguous stopping criteria.
pde2.NLtol = fluidPde.NLtol;
pde2.NLiter = fluidPde.NLiter;
pde2.matvectol = fluidPde.matvectol;

solidMeshSize = 0.0028;
[mesh2,solidMeshInfo] = mkmesh_isoqsolid(pde2,fluidMesh,solidMeshSize);
mesh2.udg = zeros(size(mesh2.dgnodes,1),3,size(mesh2.dgnodes,3));
mesh2.udg(:,1,:) = solidTemperature;

figure(4); clf; meshplot(mesh2); axis on; axis equal; axis tight;
title('Gmsh triangular solid mesh');
