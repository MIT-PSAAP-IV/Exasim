% Put the Exasim MATLAB frontend on the path. For an installed Exasim use
% run('<prefix>/share/exasim/matlab/exasim_setup.m') instead.
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

% initialize pde structure and mesh structure
[pde,~] = initializeexasim();
pde.model = "ModelD";
pde.codegenerator = "prebuilt";
pde.kernelsource = fullfile(fileparts(mfilename('fullpath')), 'canonical_kernels', 'fluid');
pde.canonicalmaster = fullfile(fileparts(mfilename('fullpath')), 'canonical_mesh', 'fluid_master.bin');
pde.modelfile = "pdemodel1";

% Choose computing platform and set number of processors
pde.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pde.mpiprocs = 4;             % number of MPI processors
pde.porder = 2;          % polynomial degree
pde.pgauss = 2*pde.porder;
pde.hybrid = 1;               % 0 -> LDG, 1 -> HDG
pde.debugmode = 0;
pde.nd = 2;
% ParaView fields from pdemodel_axialns:
%   Scalar Field 0-4 = density [kg/m^3], pressure [Pa], temperature [K], Mach, AV
%   Vector Field 0   = velocity [m/s]
pde.saveParaview = 1;
% Save Cp, Cf, and Cq from pdemodel_axialns.surfacequantities on the
% isothermal wall (boundary-condition ID 3) at face Gauss points.
pde.saveSolBouFreq = 1;
pde.ibs = 3;
pde.saveSolBouLoc = 1;

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
pde.AV = 2;
pde.AVcontinuationIter = 8;
pde.AVcontinuationLogScale = 1.5;
pde.AVcoeffStart = 0.002;
pde.AVcoeffEnd = 0.00002;
% Canonical continuation schedule shared by MATLAB, Python, and Julia.  The
% explicit values avoid libm-dependent expm1 rounding in this shock case.
pde.avparam1 = [0.002; 0.0015034374019939735; 0.0011026529166442065; ...
    0.0007791726457939881; 0.000518085978384667; ...
    0.0003073582956121101; 0.00013727624306105775; 0.0];
pde.avparam2 = [0.0; 1.3727624306105766e-06; 3.0735829561211013e-06; ...
    5.180859783846669e-06; 7.791726457939882e-06; ...
    1.1026529166442065e-05; 1.5034374019939733e-05; 2.0e-05];
pde.AVdistfunction = 1;
pde.distanceboundaryconditions = 3; % isothermal-wall flow BC tag
pde.AVsmoothingMethod = 1;
pde.AVHelmholtzCoeff = 0.0001;
AVmaxdiv = 20.0;
AVdistcoeff = nm;

% Geometric boundaries: symmetry, outflow, inflow/farfield, wall.
% Type 3 permits tangential motion; type 2 fixes both displacement components.
pde.meshadaptenabled = 0;
pde.meshadaptfield = 2;
pde.meshadaptavcomponent = 1;
pde.meshadaptalpha = 0.5;
pde.meshadaptqmin = 0.2;
pde.meshadaptqmax = 0.8;
pde.meshadaptHelmholtzCoeff = 0.001;
pde.meshadaptforcescale = 0.25;
pde.meshadaptsmoothingpasses = 30;
pde.meshadaptboundaryconditions = [3;3;3;2];

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall ...
    AVmaxdiv AVdistcoeff pde.AVcoeffStart pde.AVcoeffEnd];
pde.tau = 4.0;                  % DG stabilization parameter
pde.GMRESrestart = 500;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6; % GMRES tolerance
pde.linearsolveriter = 1000;
pde.preconditioner = 1;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;              % Newton tolerance
pde.NLiter = 10;                % Newton iterations
pde.matvectol=1e-6;             % tolerance for matrix-vector multiplication

mesh = load_isoq_canonical_mesh("fluid");
mesh.boundarycondition = [4 2 1 3]; % symmetry, outflow, inflow, wall
figure(1);clf;meshplot(mesh);

master = Master(pde);

% The canonical payload stores the wall distance and initial state so all
% three frontends serialize identical floating-point data.
dist = mesh.vdg(:,1,:);

mesh.porder = pde.porder;
mesh.xpe = master.xpe;
mesh.telem = master.telem;
figure(2); clf; scaplot(mesh,mesh.vdg(:,1,:),[],1,0); axis on; axis equal; axis tight;

TnearWall = Tinf * (Twall/Tref-1) * exp(-nm*dist) + Tinf;

figure(3); clf; scaplot(mesh,TnearWall,[],1); axis on; axis equal; axis tight;

% The coupled driver reuses this complete flow configuration and mesh, but
% postpones code generation and the solve until the solid model is ready.
if exist('isoq2d_configure_only', 'var') && isoq2d_configure_only
    return;
end

% An export-only caller can generate this exact configured case without solving.
if exist('text2code_export_directory', 'var') && ~isempty(text2code_export_directory)
    exporttext2code(pde,mesh,text2code_export_directory);
    if exist('text2code_export_only', 'var') && text2code_export_only
        return;
    end
end

pde.gencode = 1;
[sol,pde,mesh,master,dmd] = exasim(pde,mesh); %#ok<ASGLU>

vdg = getsolution('dataout/outvdg',dmd, master.npe);

figure(1); clf; meshplot(mesh,1)
figure(2); clf; scaplot(mesh, vdg(:,2,:),[],2,2);
axis equal; axis tight; colorbar;
figure(3); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1,2); colorbar;

result = postprocess_surfacequantities(pde);

figure(4); clf; plot(result.s, result.Cp, 'o-', 'LineWidth', 1.0, 'MarkerSize', 4);
grid on;
xlabel('wall arclength');
ylabel('C_p');
set(gca, 'FontSize', 16);

figure(5); clf; plot(result.s, result.Cf, 'o-', 'LineWidth', 1.0, 'MarkerSize', 4);
grid on;
xlabel('wall arclength');
ylabel('C_f');
set(gca, 'FontSize', 16);

figure(6); clf; plot(result.s, result.Cq, 'o-', 'LineWidth', 1.0, 'MarkerSize', 4);
grid on;
xlabel('wall arclength');
ylabel('C_q');
set(gca, 'FontSize', 16);
