% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

% initialize pde structure and mesh structure
[pde,mesh] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel_axialns";

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 4;             % number of MPI processors
pde.porder = 2;          % polynomial degree
pde.pgauss = 2*pde.porder;
pde.hybrid = 1;               % 0 -> LDG, 1 -> HDG
pde.debugmode = 0;
pde.nd = 2;

gam = 1.4;                      % specific heat ratio
Re = 1.56e5;                     % Reynolds number
Pr = 0.71;                      % Prandtl number
Minf = 7.6;                       % Mach number
Tref  = 266.5;
Twall = 300;
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
alpha = 0;                % angle of attack
rinf = 1.0;                     % freestream density
ruinf = cos(alpha);             % freestream horizontal velocity
rvinf = sin(alpha);             % freestream vertical velocity
pinf = 1/(gam*Minf^2);          % freestream pressure
rEinf = 0.5+pinf/(gam-1);       % freestream energy

nm = 1e2;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall 0.0015 nm];
pde.tau = 8.0;                  % DG stabilization parameter
pde.GMRESrestart = 250;         %try 50
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6; % GMRES tolerance
pde.linearsolveriter = 500; %try 100
pde.preconditioner = 1;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;              % Newton tolerance
pde.NLiter = 10;                % Newton iterations
pde.matvectol=1e-6;             % tolerance for matrix-vector multiplication

mesh = mkmesh_isoq2d(pde.porder, 5e-4);
mesh.boundarycondition = [5 2 1 3]; % symmetry, outflow, inflow, wall
figure(1);clf;meshplot(mesh);

master = Master(pde);

% initial artificial viscosity
dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,[4]); % distance to the wall
mesh.vdg(:,1,:) = dist;

mesh.porder = pde.porder;
mesh.xpe = master.xpe;
mesh.telem = master.telem;
figure(2); clf; scaplot(mesh,mesh.vdg(:,1,:),[],1,0); axis on; axis equal; axis tight;

% intial solution
ui = [rinf ruinf rvinf rEinf];
UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4),0,0,0,0,0,0,0,0}); % freestream
UDG(:,2,:) = UDG(:,2,:).*tanh(nm*dist);
UDG(:,3,:) = UDG(:,3,:).*tanh(nm*dist);
TnearWall = Tinf * (Twall/Tref-1) * exp(-nm*dist) + Tinf;
UDG(:,4,:) = TnearWall + 0.5*(UDG(:,2,:).*UDG(:,2,:) + UDG(:,3,:).*UDG(:,3,:));
mesh.udg = UDG;

figure(3); clf; scaplot(mesh,TnearWall,[],1); axis on; axis equal; axis tight;

avparam = [0.002 0.0011 0.0008 0.0005];
for i = 1:length(avparam)
  pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall avparam(i) nm];
  if (i==1)
    pde.gencode = 1;
  else
    pde.gencode = 0;
  end
  [sol,pde,mesh,master,dmd] = exasim(pde,mesh);
  mesh.udg = sol;
  figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1,1); colorbar;
  figure(2); clf; scaplot(mesh, mesh.vdg,[],1); colorbar;
end
