% Export the isoq physics-parameter sweep as a Text2Code application.
%
% This script mirrors pdeapp_sweep.m through mesh/state construction, then
% writes a self-contained Text2Code package under apps/navierstokes/isoq2d_sweep.

sourceScript = fullfile(fileparts(mfilename('fullpath')), 'pdeapp_sweep.m');
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

exasimroot = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..');
exportdir = fullfile(exasimroot, 'apps', 'navierstokes', 'isoq2d_sweep');
if exist(exportdir, 'dir') == 0
    mkdir(exportdir);
end

% Keep the exported case reproducible by removing stale Text2Code inputs and
% generated run directories before writing a fresh package.
staleFiles = ["README.md", "pdeapp.txt", "pdemodel.txt", "grid.bin", ...
              "xdg.bin", "udg.bin", "vdg.bin", "wdg.bin"];
for i = 1:numel(staleFiles)
    fn = fullfile(exportdir, staleFiles(i));
    if exist(fn, 'file')
        delete(fn);
    end
end
for d = ["datain", "dataout"]
    dn = fullfile(exportdir, d);
    if exist(dn, 'dir')
        rmdir(dn, 's');
    end
end

% initialize pde structure and mesh structure
[pde,mesh] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel_axialns2";

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 4;             % number of MPI processors
pde.porder = 2;               % polynomial degree
pde.pgauss = 2*pde.porder;
pde.hybrid = 1;               % 0 -> LDG, 1 -> HDG
pde.debugmode = 0;
pde.nd = 2;

gam = 1.4;                    % specific heat ratio
Re = 1.835e5;                 % Reynolds number
Pr = 0.71;                    % Prandtl number
Minf = 7;                     % Mach number
Tref  = 124.49;
Twall = 294.44;
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
alpha = 0;                    % angle of attack
rinf = 1.0;                   % freestream density
ruinf = cos(alpha);           % freestream horizontal velocity
rvinf = sin(alpha);           % freestream vertical velocity
pinf = 1/(gam*Minf^2);        % freestream pressure
rEinf = 0.5+pinf/(gam-1);     % freestream energy

nm = 1e2;
baselineAV = 0.0015;
basePhysicsParam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall baselineAV nm];
pde.physicsparam = basePhysicsParam;

% Sweep the artificial-viscosity parameter around the pdeapp_sweep baseline.
% Warm-starting lets each case after the first start from the previous case's
% converged solution.
avParamSweep = [0.0015; 0.0011; 0.0008];
pde.physicsparamsweep = repmat(basePhysicsParam, numel(avParamSweep), 1);
pde.physicsparamsweep(:,12) = avParamSweep;
pde.physicsparamwarmstart = 1;

pde.tau = 8.0;                % DG stabilization parameter
pde.GMRESrestart = 250;       %try 50
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6;   % GMRES tolerance
pde.linearsolveriter = 500;   %try 100
pde.preconditioner = 1;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;             % Newton tolerance
pde.NLiter = 10;              % Newton iterations
pde.matvectol=1e-6;           % tolerance for matrix-vector multiplication
pde.gencode = 1;

mesh = mkmesh_isoq2d4(pde.porder, 2e-3);
mesh.boundarycondition = [5 2 1 3]; % symmetry, outflow, inflow, wall

% Convert closure-based boundary expressions to numeric strings so pdeapp.txt
% is standalone and does not depend on MATLAB workspace variables.
deltay = min(mesh.p(2,:));
L = max(mesh.p(1,:));
mesh.boundaryexpr = [ ...
    "abs(y-(" + num2str(deltay, 17) + "))<1e-6", ...
    "x>(" + num2str(L, 17) + ")-1e-4", ...
    "((x<-1e-3)||(y>0.1))", ...
    "abs(x)<20+1e-6"];

master = Master(pde);

% initial artificial viscosity
dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,[4]); % distance to the wall
mesh.vdg(:,1,:) = dist;

mesh.porder = pde.porder;
mesh.xpe = master.xpe;
mesh.telem = master.telem;

% intial solution
ui = [rinf ruinf rvinf rEinf];
UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4),0,0,0,0,0,0,0,0}); % freestream
UDG(:,2,:) = UDG(:,2,:).*tanh(nm*dist);
UDG(:,3,:) = UDG(:,3,:).*tanh(nm*dist);
TnearWall = Tinf * (Twall/Tref-1) * exp(-nm*dist) + Tinf;
UDG(:,4,:) = TnearWall + 0.5*(UDG(:,2,:).*UDG(:,2,:) + UDG(:,3,:).*UDG(:,3,:));
mesh.udg = UDG;

exporttext2code(pde, mesh, exportdir);

requiredFiles = ["pdeapp.txt", "pdemodel.txt", "grid.bin", "xdg.bin", "udg.bin", "vdg.bin"];
for i = 1:numel(requiredFiles)
    fn = fullfile(exportdir, requiredFiles(i));
    if exist(fn, 'file') == 0
        error("Missing exported Text2Code file: %s", fn);
    end
end

fprintf("Exported %s from %s\n", exportdir, sourceScript);
