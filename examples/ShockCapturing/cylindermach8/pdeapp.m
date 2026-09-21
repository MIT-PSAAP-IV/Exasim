% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();

% Define a PDE model: governing equations, initial solutions, and boundary conditions
pde.model = "ModelD";          % ModelC, ModelD, ModelW
pde.modelfile = "pdemodel";    % name of a file defining the PDE model

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 1;              % number of MPI processors
pde.hybrid = 1;
pde.porder = 4;          % polynomial degree

mesh = mkmesh_cyl(pde.porder);
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
pde.AVcontinuationLogScale = 1;
pde.AVcoeffStart = 0.060;
pde.AVcoeffEnd =  0.015;
pde.AVdistfunction = 1;
pde.AVsmoothingMethod = 1; % 0: legacy DG2CG2, 1: internal HDG Helmholtz filter
pde.AVHelmholtzCoeff = 0.025; % multiplier for sqrt(smoothed nodal Jacobian)

AVmaxdiv = 2.0;
AVdistcoeff = 10;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref ...
    Twall AVmaxdiv AVdistcoeff pde.AVcoeffStart pde.AVcoeffEnd];

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

% An export-only caller can set these workspace variables before running
% this script.  The text2code package is then generated from this exact
% configured PDE and mesh, without solving or manually recreating inputs.
if exist('text2code_export_directory', 'var') && ~isempty(text2code_export_directory)
    pde.saveParaview = 1;
    exporttext2code(pde, mesh, text2code_export_directory);
    if exist('text2code_export_only', 'var') && text2code_export_only
        return;
    end
end

[sol,pde,mesh,master,dmd] = exasim(pde,mesh);

vdg = getsolution('dataout/outvdg',dmd,25);
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],2,2);
axis equal; axis tight; colorbar;

param = pde.physicsparam;
figure(2); clf; scaplot(mesh, (param(end)*vdg(:,2,:)).*tanh(param(end-2)*vdg(:,1,:)), [], 2);
axis equal; axis tight; colorbar;

figure(3); clf; scaplot(mesh, vdg(:,2,:), [], 2);
axis equal; axis tight; colorbar;
