% Configure the axisymmetric solid heat equation on a Gmsh triangle mesh.
if ~exist('fluidMesh','var') || ~exist('fluidPde','var')
    error('pdeapp2 requires fluidMesh and fluidPde from pdeapp1.');
end

[pde2,~] = initializeexasim();
pde2.model = "ModelD";
pde2.codegenerator = "prebuilt";
pde2.kernelsource = fullfile(fileparts(mfilename('fullpath')), 'canonical_kernels', 'solid');
pde2.canonicalmaster = fullfile(fileparts(mfilename('fullpath')), 'canonical_mesh', 'solid_master.bin');
pde2.modelfile = "pdemodel2";
pde2.platform = fluidPde.platform;
pde2.mpiprocs = 2;
pde2.hybrid = 1;
pde2.debugmode = fluidPde.debugmode;
pde2.nd = 2;
pde2.gencode = 1;
% Use the shared backend partitioner in every frontend.  The legacy frontend
% face partitioners are not byte-identical for this two-rank triangle mesh.
pde2.Cxxpreprocessing = 1;

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

mesh2 = load_isoq_canonical_mesh("solid");
solidMeshInfo.triangles = size(mesh2.t,2);

figure(4); clf; meshplot(mesh2); axis on; axis equal; axis tight;
title('Gmsh triangular solid mesh');
