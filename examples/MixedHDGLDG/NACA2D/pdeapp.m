% Steady NACA0012: one HDG wall layer and LDG in the outer flow.
caseDir = fileparts(mfilename('fullpath'));
if exist('initializeexasim','file') ~= 2
    setup = fullfile(caseDir,'..','..','..','frontends','Matlab','exasim_setup.m');
    if ~isfile(setup)
        error('Run your installed share/exasim/matlab/exasim_setup.m first.');
    end
    run(setup);
end
addpath(caseDir);
[pde,mesh] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.platform = "cpu";
pde.mpiprocs = 2;
pde.ncu = 4;
pde.porder = 4;
pde.pgauss = 8;
% gamma, Reynolds, Prandtl, Mach, and freestream conservative state.
pde.physicsparam = [1.4,100,0.72,0.2,1,1,0,45.1429];
pde.tau = 3;
pde.NLtol = 1e-6;
pde.NLiter = 20;
pde.linearsolvertol = 1e-7;
pde.linearsolveriter = 400;
pde.GMRESrestart = 400;
pde.GMRESortho = 0;
pde.matvectol = 1e-3;
pde.preconditioner = 1;
pde.ppdegree = 0;
pde.RBdim = 0;

% The unchanged donor mesh is supplied locally, so copied cases also work.
[mesh.p,mesh.t] = readmesh(fullfile(caseDir,'grid.bin'),0);
mesh.boundaryexpr = {@(p) sqrt((p(1,:)-0.5).^2+p(2,:).^2)<3, ...
    @(p) abs(p(1,:))<20};
mesh.boundarycondition = [1;2]; % no-slip adiabatic wall, far field
mesh.curvedboundary = [1;0];
mesh.curvedboundaryexpr = {@(p) p(2,:).^2- ...
    (5*0.01*12*(0.29690*sqrt(abs(p(1,:)))-0.12600*p(1,:)- ...
    0.35160*p(1,:).^2+0.28430*p(1,:).^3-0.10150*p(1,:).^4)).^2, ...
    @(p) 0*p(1,:)};

mesh = mkmesh_naca0012(pde.porder,1,1);  

% Selection uses a BC code, not the index in boundaryexpr.
selection = struct('seed_boundaries',1,'layers',1);
if ~isempty(getenv('HDGLDG_RUN_DIR'))
    pde.datapath = string(getenv('HDGLDG_RUN_DIR'));
    pde.builddir = fullfile(pde.datapath,'.exasim');
end
[sol,pde,mesh] = hdgldgsolve(pde,mesh,selection,1);

mesh.porder=pde.porder;
[mesh.f, mesh.tprd, t2t] = facenumbering(mesh.p,mesh.t,1,mesh.boundaryexpr,mesh.periodicexpr);
[mesh.dgnodes,mesh.elemtype]=createdgnodes(mesh.p,mesh.t,mesh.f,mesh.curvedboundary,mesh.curvedboundaryexpr,pde.porder);
[mesh.xpe,mesh.telem]=masternodes(pde.porder,size(mesh.p,1),mesh.elemtype);

mesh.elemtype = 1;
figure(1); clf; scaplot(mesh,eulereval(sol(:,1:4,:),'M',gam),[],2,2); 
axis on; axis equal; axis tight;
axis([-0.5 2 -0.62 0.62])
set(gca,'fontsize', 16);
hold on;
x = mesh.dgnodes(:,1,:);
y = mesh.dgnodes(:,2,:);
plot(x(:), y(:), 'o');
