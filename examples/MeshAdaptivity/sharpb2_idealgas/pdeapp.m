% Sharp-B axisymmetric ideal-gas flow with backend AV and mesh adaptation.

caseDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(fileparts(caseDir)));
run(fullfile(repoRoot,'frontends','Matlab','exasim_setup.m'));

[pde,mesh] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel_axialns";
pde.platform = "cpu";
pde.mpiprocs = 8;
pde.porder = 2;
pde.pgauss = 2*pde.porder;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 2;
pde.saveParaview = 1;
pde.datapath = caseDir;
pde.builddir = fullfile(caseDir,'.exasim');
pde.buildpath = pde.builddir;

% Preserve the flow conditions from examples/NavierStokes/sharpb2.
gam = 1.4;
Re = 9.84e5;
Pr = 0.71;
Minf = 21.38;
Tref = 260.6;
Twall = 1400;
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
alpha = 0;
rinf = 1.0;
ruinf = cos(alpha);
rvinf = sin(alpha);
rEinf = 0.5+pinf/(gam-1);

% Use the backend AV continuation and mesh-adaptation workflow from
% examples/MeshAdaptivity/isoq2d_idealgas.  The AV magnitudes follow the
% more strongly stabilized Mach-21 Sharp-B continuation.
nm = 1e2;
pde.AV = 1;
pde.AVcontinuationIter = 9;
pde.AVcontinuationLogScale = 2;
pde.AVcoeffStart = 0.005;
pde.AVcoeffEnd = 0.000016;
pde.AVdistfunction = 1;
pde.distanceboundaryconditions = 3; % isothermal-wall flow BC tag
pde.AVsmoothingMethod = 1;
pde.AVHelmholtzCoeff = 0.001;
AVmaxdiv = 60.0;
AVdistcoeff = nm;

pde.meshadaptenabled = 1;
pde.meshadaptfield = 2; % physical pressure from visscalars
pde.meshadaptavcomponent = 1;
pde.meshadaptalpha = 0.5;
pde.meshadaptqmin = 0.2;
pde.meshadaptqmax = 0.8;
pde.meshadaptHelmholtzCoeff = 0.001;
pde.meshadaptforcescale = 0.25;
pde.meshadaptsmoothingpasses = 30;
% Geometric boundaries: axis, lower farfield, upper farfield, wall, outflow.
% Type 3 permits tangential motion; type 2 fixes both displacement components.
pde.meshadaptboundaryconditions = [3;3;3;2;3];

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall ...
    AVmaxdiv AVdistcoeff pde.AVcoeffStart pde.AVcoeffEnd];
pde.tau = 4.0;
pde.GMRESrestart = 250;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6;
pde.linearsolveriter = 500;
pde.preconditioner = 1;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;
pde.NLiter = 10;
pde.matvectol = 1e-6;

mesh = mkmesh_sharpb2(pde.porder);
mesh.boundarycondition = [5 1 1 3 2]; % symmetry, inflow, inflow, wall, outflow
figure(1); clf; meshplot(mesh); axis equal; axis tight;

master = Master(pde);
dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,4);
mesh.vdg = zeros(size(mesh.dgnodes,1),2,size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = dist;

mesh.porder = pde.porder;
mesh.xpe = master.xpe;
mesh.telem = master.telem;

ui = [rinf ruinf rvinf rEinf];
UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4),0,0,0,0,0,0,0,0});
UDG(:,2,:) = UDG(:,2,:).*tanh(nm*dist);
UDG(:,3,:) = UDG(:,3,:).*tanh(nm*dist);
TnearWall = Tinf*(Twall/Tref-1)*exp(-nm*dist)+Tinf;
UDG(:,4,:) = TnearWall + 0.5*(UDG(:,2,:).^2+UDG(:,3,:).^2);
mesh.udg = UDG;

pde.gencode = 1;
[sol,pde,mesh,master,dmd] = exasim(pde,mesh); %#ok<ASGLU>

xdg = getsolution(fullfile(caseDir,'dataout','outxdg'),dmd,master.npe);
vdg = getsolution(fullfile(caseDir,'dataout','outvdg'),dmd,master.npe);
adaptedMesh = mesh;
adaptedMesh.dgnodes = xdg;

figure(1); clf; meshplot(adaptedMesh,1); axis equal; axis tight;
figure(2); clf; scaplot(adaptedMesh,vdg(:,2,:),[],2,2);
axis equal; axis tight; colorbar;
figure(3); clf; scaplot(adaptedMesh,eulereval(sol,'M',gam,Minf),[0 Minf],1,1);
axis equal; axis tight; colorbar;
