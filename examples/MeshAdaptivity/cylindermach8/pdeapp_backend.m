% Backend AV continuation and mesh-adaptivity counterpart of pdeapp_frontend.m.
casepath = fileparts(mfilename('fullpath'));
run(fullfile(casepath, '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

[pde,~] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.platform = "cpu";
pde.mpiprocs = 4;
pde.hybrid = 1;
pde.porder = 2;

rundir = fullfile(casepath, 'backend_run');
if ~isfolder(rundir), mkdir(rundir); end
pde.datapath = string(rundir);
pde.builddir = string(fullfile(rundir, '.exasim'));
pde.buildpath = pde.builddir;

mesh = mkmesh_square(51,32,pde.porder,1,1,1,1,1);
mesh.p(1,:) = logdec(mesh.p(1,:), 3);
mesh.dgnodes(:,1,:) = logdec(mesh.dgnodes(:,1,:), 3);
mesh = mkmesh_halfcircle(mesh, 1, 3, 4.0, pi/2, 3*pi/2);
mesh.porder = pde.porder;
mesh.boundaryexpr = {@(p) sqrt(p(1,:).^2+p(2,:).^2)<1+1e-6, ...
                     @(p) p(1,:)>-1e-7, @(p) abs(p(1,:))<20};
mesh.periodicexpr = {};
mesh.boundarycondition = [3;6;5];

gam = 1.4; Re = 1.835e5; Pr = 0.71; Minf = 8.03;
Tref = 265; Twall = 300;
pinf = 1/(gam*Minf^2); Tinf = pinf/(gam-1);
angleOfAttack = 0; rinf = 1.0;
ruinf = cos(angleOfAttack); rvinf = sin(angleOfAttack);
rEinf = 0.5+pinf/(gam-1);

pde.tau = 1.0;
pde.GMRESrestart = 200;
pde.linearsolvertol = 1e-6;
pde.linearsolveriter = 200;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;
pde.NLiter = 10;
pde.matvectol = 1e-6;

pde.AV = 1;
pde.AVcontinuationIter = 10;
pde.AVcontinuationLogScale = 1.5;
pde.AVcoeffStart = 0.060;
pde.AVcoeffEnd = 0.01;
pde.AVdistfunction = 1;
pde.distanceboundaryconditions = [3]; % boundary-condition IDs stored in backend mesh.bf
pde.AVsmoothingMethod = 1;
pde.AVHelmholtzCoeff = 0.025;
AVmaxdiv = 2.0; AVdistcoeff = 100;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref ...
    Twall AVmaxdiv AVdistcoeff pde.AVcoeffStart pde.AVcoeffEnd];

% Match meshadapt2d(mesh,...,qmin,qmax,diffcoeff,bcs,params,ndg2cg).
pde.meshadaptenabled = 1;
pde.meshadaptfield = 3;
pde.meshadaptalpha = 0.5;
pde.meshadaptHelmholtzCoeff = 5e-2;
pde.meshadaptforcescale = 0.2; % params(3) in pdeapp_frontend.m
pde.meshadaptsmoothingpasses = 30;
pde.meshadaptboundaryconditions = [2;3;3];

mesh.f = facenumbering(mesh.p,mesh.t,pde.elemtype,mesh.boundaryexpr,mesh.periodicexpr);
dist = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,[1]);
mesh.vdg = zeros(size(mesh.dgnodes,1),2,size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = dist;
ui = [rinf ruinf rvinf rEinf];
UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4)});
UDG(:,2,:) = UDG(:,2,:).*tanh(10*dist);
UDG(:,3,:) = UDG(:,3,:).*tanh(10*dist);
TnearWall = Tinf * (Twall/Tref-1) * exp(-10*dist) + Tinf;
UDG(:,4,:) = TnearWall + 0.5*(UDG(:,2,:).^2 + UDG(:,3,:).^2);
mesh.udg = UDG;

% An export-only caller can generate this exact configured case without solving.
if exist('text2code_export_directory', 'var') && ~isempty(text2code_export_directory)
    exporttext2code(pde,mesh,text2code_export_directory);
    if exist('text2code_export_only', 'var') && text2code_export_only
        return;
    end
end

setenv('EXASIM_MESHADAPT_VERIFY','0');
[sol,pde,mesh,master,dmd] = exasim(pde,mesh); %#ok<ASGLU>
