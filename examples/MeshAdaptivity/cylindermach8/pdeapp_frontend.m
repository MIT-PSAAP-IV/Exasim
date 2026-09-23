% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', ...
    'frontends', 'Matlab', 'Mesh', 'adaptivity'));

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();

casepath = fileparts(mfilename('fullpath'));
rundir = fullfile(casepath, 'frontend_run');
if ~isfolder(rundir), mkdir(rundir); end
pde.datapath = string(rundir);
pde.builddir = string(fullfile(rundir, '.exasim'));
pde.buildpath = pde.builddir;

% Define a PDE model: governing equations, initial solutions, and boundary conditions
pde.model = "ModelD";          % ModelC, ModelD, ModelW
pde.modelfile = "pdemodel";    % name of a file defining the PDE model

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 1;              % number of MPI processors
pde.hybrid = 1;
pde.porder = 2;          % polynomial degree

%mesh = mkmesh_cyl(pde.porder);
mesh = mkmesh_square(51,32,pde.porder,1,1,1,1,1);
mesh.p(1,:) = logdec(mesh.p(1,:), 3);
mesh.dgnodes(:,1,:) = logdec(mesh.dgnodes(:,1,:), 3);
mesh = mkmesh_halfcircle(mesh, 1, 3, 4.0, pi/2, 3*pi/2);
mesh.porder = pde.porder;
mesh.boundaryexpr = {@(p) sqrt(p(1,:).^2+p(2,:).^2)<1+1e-6, @(p) p(1,:)>-1e-7, @(p) abs(p(1,:))<20};
mesh.periodicexpr = {};
% iso-thermal wall, supersonic outflow, supersonic inflow
mesh.boundarycondition = [3;6;5];

gam = 1.4;                      % specific heat ratio
Re = 1.835e5;                     % Reynolds number
Pr = 0.71;                      % Prandtl number
Minf = 8.03;                     % Mach number
Tref  = 265;
Twall = 300;
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
alpha = 0;                % angle of attack
rinf = 1.0;                     % freestream density
ruinf = cos(alpha);             % freestream horizontal velocity
rvinf = sin(alpha);             % freestream vertical velocity
pinf = 1/(gam*Minf^2);          % freestream pressure
rEinf = 0.5+pinf/(gam-1);       % freestream energy

% solver parameters
pde.tau = 1.0;                  % DG stabilization parameter
pde.GMRESrestart = 200;         %try 50
pde.linearsolvertol = 1e-6; % GMRES tolerance
pde.linearsolveriter = 200; %try 100
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;              % Newton tolerance
pde.NLiter = 10;                 % Newton iterations
pde.matvectol=1e-6;             % tolerance for matrix-vector multiplication

% AV parameters
pde.AV = 1;
pde.AVcontinuationIter = 10;
pde.AVcontinuationLogScale = 1.5;
pde.AVcoeffStart = 0.060;
pde.AVcoeffEnd =  0.01;
pde.AVdistfunction = 0;
pde.AVsmoothingMethod = 1; % 0: legacy DG2CG2, 1: internal HDG Helmholtz filter
pde.AVHelmholtzCoeff = 0.025; % multiplier for sqrt(smoothed nodal Jacobian)

AVmaxdiv = 2.0;
AVdistcoeff = 100;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref ...
    Twall AVmaxdiv AVdistcoeff pde.AVcoeffStart pde.AVcoeffEnd];

% mesh adaptivity parameters
qmin = 0.2; qmax = 0.8; diffcoeff = 5e-2;
bcs = [2;3;3];
params = [1 1 0.2];
ndg2cg = 30;

% wall distance function
mesh.f = facenumbering(mesh.p,mesh.t,pde.elemtype,mesh.boundaryexpr,mesh.periodicexpr);
dist = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,[1]); % distance to the wall
mesh.vdg = zeros(size(mesh.dgnodes,1),2,size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = dist;
mesh.vdg(:,2,:) = 0; % reserved for AV

% intial solution
ui = [rinf ruinf rvinf rEinf];
UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4)}); % freestream
UDG(:,2,:) = UDG(:,2,:).*tanh(10*dist);
UDG(:,3,:) = UDG(:,3,:).*tanh(10*dist);
TnearWall = Tinf * (Twall/Tref-1) * exp(-10*dist) + Tinf;
UDG(:,4,:) = TnearWall + 0.5*(UDG(:,2,:).*UDG(:,2,:) + UDG(:,3,:).*UDG(:,3,:));
mesh.udg = UDG;

[pde,mesh,master,dmd] = preprocessing(pde,mesh);
[~,cgelcon,rowent2elem,colent2elem,cgent2dgent] = mkcgent2dgent(mesh.dgnodes,1e-8);

c1 = loginc(linspace(pde.AVcoeffStart, 0, pde.AVcontinuationIter),pde.AVcontinuationLogScale);
c2 = loginc(linspace(0, pde.AVcoeffEnd, pde.AVcontinuationIter),pde.AVcontinuationLogScale);
for i = 1:pde.AVcontinuationIter
  pde.physicsparam(end-1:end) = [c1(i) c2(i)];
  mesh.vdg(:,1,:) = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,[1]); % distance to the wall
  writeverify(rundir,i,'wall_distance',mesh.vdg(:,1,:));
  if i>1
    div = getavfield(mesh.udg(:,1:4,:),mesh.udg(:,5:end,:),mesh.vdg(:,1,:),pde.physicsparam);
  else
    div = 0*mesh.vdg(:,1,:);
  end
  writeverify(rundir,i,'av_raw',div);
  solhm = pdeapp_hm2d(mesh, master, div, pde.AVHelmholtzCoeff);
  mesh.vdg(:,2,:) = solhm(:,1,:);
  writeverify(rundir,i,'av_smoothed',mesh.vdg(:,2,:));
  av = (pde.physicsparam(end-1) + pde.physicsparam(end)*mesh.vdg(:,2,:)).*tanh(pde.physicsparam(end-2)*mesh.vdg(:,1,:));

  figure(1); clf; scaplot(mesh, av, [], 2, 2); axis equal; axis tight; colorbar;

  [pde,mesh,master,dmd] = preprocessing(pde,mesh);
  if i == 1
    kkgencode(pde);
    compilerstr = cmakecompile(pde); % use cmake to compile C++ source codes
  end
  runcode(pde, 1);
  sol = fetchsolution(pde,master,dmd, pde.datapath + "/dataout" + model_strn(pde));
  mesh.udg = sol;
  writeverify(rundir,i,'flow_solution',sol);

  figure(2); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],2,2);
  axis equal; axis tight; colorbar;

  if i < pde.AVcontinuationIter
    div = getavfield(sol(:,1:4,:),sol(:,5:end,:),mesh.vdg(:,1,:),pde.physicsparam);
    sensor_scalar = eulereval(sol,'p',gam,Minf);
    [sensor_raw,~] = discontinuitysensor(master,mesh,sensor_scalar);
    p = indicatorfield(mesh, master, sensor_scalar);
    field1 = div/max(abs(div(:)));
    field2 = p/max(abs(p(:)));
    eta = 0.5*field1 + 0.5*field2;
    writeverify(rundir,i,'sensor_scalar',sensor_scalar);
    writeverify(rundir,i,'sensor_raw',sensor_raw);
    writeverify(rundir,i,'field1',field1);
    writeverify(rundir,i,'field2',field2);
    writeverify(rundir,i,'eta',eta);
    figure(3); clf; scaplot(mesh, eta, [], 2, 2); axis equal; axis tight; colorbar;

    [~,hmin,hmax] = meshsizefield(mesh,master,qmin,qmax);
    [solle,mu,lambda,fx,fy,h,solhm] = meshadapt2d(mesh, master, cgelcon, colent2elem, rowent2elem, cgent2dgent, eta, qmin, qmax, diffcoeff, bcs, params, ndg2cg);
    writeverify(rundir,i,'displacement',solle(:,1:2,:));
    mesh.dgnodes = mesh.dgnodes + solle(:,1:2,:);
    writeverify(rundir,i,'xdg',mesh.dgnodes);

    figure(4); clf; meshplot(mesh,1); axis equal; axis tight;
  end
end

function writeverify(rundir,iteration,name,values)
verificationDirectory = fullfile(rundir,'verification');
if ~isfolder(verificationDirectory), mkdir(verificationDirectory); end
filename = fullfile(verificationDirectory,sprintf('aviter%d_%s.bin',iteration,name));
fileID = fopen(filename,'w');
fwrite(fileID,values(:),'double');
fclose(fileID);
end
