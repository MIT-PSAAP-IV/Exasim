% Verify backend mesh adaptivity against pdeapp_meshadapt.m.
% Run pdeapp_meshadapt first to create meshadapt_reference.mat, then run this
% application. All mesh movement in this file is performed by the backend.

casepath = fileparts(mfilename('fullpath'));
run(fullfile(casepath, '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));
addpath(casepath);

[pde,~] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.platform = "cpu";
pde.mpiprocs = 1;
pde.hybrid = 1;
pde.porder = 2;

% Keep this verification run separate from the MATLAB-reference outputs.
rundir = fullfile(casepath, 'backendmeshadapt_run');
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

gam = 1.4;
Re = 1.835e5;
Pr = 0.71;
Minf = 8.03;
Tref = 265;
Twall = 300;
pinf = 1/(gam*Minf^2);
Tinf = pinf/(gam-1);
angleOfAttack = 0;
rinf = 1.0;
ruinf = cos(angleOfAttack);
rvinf = sin(angleOfAttack);
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
pde.AVcontinuationLogScale = 1;
pde.AVcoeffStart = 0.060;
pde.AVcoeffEnd = 0.015;
pde.AVdistfunction = 1;
pde.AVsmoothingMethod = 1;
pde.AVHelmholtzCoeff = 0.012;

AVmaxdiv = 2.0;
AVdistcoeff = 100;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref ...
    Twall AVmaxdiv AVdistcoeff pde.AVcoeffStart pde.AVcoeffEnd];

mesh.f = facenumbering(mesh.p,mesh.t,pde.elemtype,mesh.boundaryexpr,mesh.periodicexpr);
dist = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,1);
mesh.vdg = zeros(size(mesh.dgnodes,1),2,size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = dist;
mesh.vdg(:,2,:) = 0;

ui = [rinf ruinf rvinf rEinf];
UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4)});
UDG(:,2,:) = UDG(:,2,:).*tanh(10*dist);
UDG(:,3,:) = UDG(:,3,:).*tanh(10*dist);
TnearWall = Tinf * (Twall/Tref-1) * exp(-10*dist) + Tinf;
UDG(:,4,:) = TnearWall + 0.5*(UDG(:,2,:).^2 + UDG(:,3,:).^2);
mesh.udg = UDG;

% These values exactly match pdeapp_meshadapt.m and meshadapt2d.m.
pde.meshadaptenabled = 1;
pde.meshadaptfield = 3;                 % pressure from VisScalars
pde.meshadaptavcomponent = 1;           % vdg(:,2,:) / stored AV component
pde.meshadaptalpha = 0.25;
pde.meshadaptqmin = 0.2;
pde.meshadaptqmax = 0.8;
pde.meshadaptHelmholtzCoeff = 2e-2;
pde.meshadapttargetexponent = 2.0;
pde.meshadaptpoissonratio = 0.2;
pde.meshadaptyoungmodulus = 1.0;
pde.meshadaptminimumyoungmodulus = 1e-3;
pde.meshadaptshearscale = 1.0;
pde.meshadaptvolumetricscale = 1.0;
pde.meshadaptforcescale = 1.0;
pde.meshadaptsmoothingpasses = 30;
pde.meshadaptiterations = 6;
pde.meshadaptHelmholtzTau = 2.0;
pde.meshadaptelasticitytau = 1e3;
adaptationBoundaryConditions = [2;3;3];
pde.meshadaptboundaryconditions = zeros(max(mesh.boundarycondition),1);
pde.meshadaptboundaryconditions(mesh.boundarycondition) = adaptationBoundaryConditions;

% MATLAB has no backtracking. beta=1 reproduces its full update whenever the
% proposed mesh is valid; the Jacobian threshold only rejects invalid meshes.
pde.meshadaptdamping = 1.0;
pde.meshadaptminimumjacobianratio = 1e-8;

oldVerification = getenv('EXASIM_MESHADAPT_VERIFY');
restoreEnvironment = onCleanup(@() setenv('EXASIM_MESHADAPT_VERIFY',oldVerification));
setenv('EXASIM_MESHADAPT_VERIFY','1');
[sol,pde,mesh,master,dmd] = exasim(pde,mesh); %#ok<ASGLU>

npe = size(mesh.dgnodes,1);
ne = size(mesh.dgnodes,3);
nd = 2;
prefix = fullfile(rundir,'dataout','out_meshadapt_');
backend = struct();
backend.sensor_scalar = readBackendField(prefix,'sensor_scalar',[npe 1 ne]);
backend.sensor_raw = readBackendField(prefix,'sensor_raw',[npe 1 ne]);
backend.field1 = readBackendField(prefix,'field1',[npe 1 ne]);
backend.field2 = readBackendField(prefix,'field2',[npe 1 ne]);
backend.eta = readBackendField(prefix,'eta',[npe 1 ne]);
backend.history = struct([]);
for iteration = 1:pde.meshadaptiterations
    tag = "iter" + string(iteration) + "_";
    backend.history(iteration).h = readBackendField(prefix,tag+"h",[npe 1 ne]);
    backend.history(iteration).hmin = readBackendField(prefix,tag+"hmin",[1 1 1]);
    backend.history(iteration).hmax = readBackendField(prefix,tag+"hmax",[1 1 1]);
    backend.history(iteration).mu = readBackendField(prefix,tag+"mu",[npe 1 ne]);
    backend.history(iteration).lambda = readBackendField(prefix,tag+"lambda",[npe 1 ne]);
    backend.history(iteration).helmholtz = readBackendField(prefix,tag+"helmholtz",[npe 1+nd ne]);
    backend.history(iteration).force = readBackendField(prefix,tag+"force",[npe nd ne]);
    backend.history(iteration).displacement = readBackendField(prefix,tag+"displacement",[npe nd ne]);
    backend.history(iteration).xdg = readBackendField(prefix,tag+"xdg",size(mesh.dgnodes));
end
backend.xdg = backend.history(end).xdg;
save(fullfile(casepath,'meshadapt_backend.mat'),'backend','-v7.3');

referenceFile = fullfile(casepath,'meshadapt_reference.mat');
if isfile(referenceFile)
    referenceData = load(referenceFile,'meshadapt_reference','meshadapt_history');
    fprintf('\nBackend mesh-adaptivity comparison against pdeapp_meshadapt.m\n');
    compareField('sensor scalar',backend.sensor_scalar,referenceData.meshadapt_reference.sensor_scalar);
    compareField('raw sensor',backend.sensor_raw,referenceData.meshadapt_reference.sensor_raw);
    compareField('field1',backend.field1,referenceData.meshadapt_reference.field1);
    compareField('field2',backend.field2,referenceData.meshadapt_reference.field2);
    compareField('eta',backend.eta,referenceData.meshadapt_reference.eta);
    for iteration = 1:pde.meshadaptiterations
        fprintf('Iteration %d\n',iteration);
        names = {'h','hmin','hmax','mu','lambda','helmholtz','force','displacement','xdg'};
        for i = 1:numel(names)
            name = names{i};
            compareField(name,backend.history(iteration).(name), ...
                         referenceData.meshadapt_history(iteration).(name));
        end
    end
else
    warning(['Reference file not found. Run pdeapp_meshadapt.m first; backend ' ...
             'results were saved to %s.'],fullfile(casepath,'meshadapt_backend.mat'));
end

adaptedMesh = mesh;
adaptedMesh.dgnodes = backend.xdg;
figure(4); clf; meshplot(adaptedMesh,1); axis on; axis equal; axis tight;

function value = readBackendField(prefix,name,shape)
filename = char(string(prefix) + string(name) + "_np0.bin");
fileID = fopen(filename,'r');
if fileID < 0, error('Cannot open backend mesh-adaptivity output %s.',filename); end
cleaner = onCleanup(@() fclose(fileID));
value = fread(fileID,prod(shape),'double');
if numel(value) ~= prod(shape)
    error('Unexpected size in %s: got %d values, expected %d.', ...
          filename,numel(value),prod(shape));
end
value = reshape(value,shape);
end

function compareField(name,computed,reference)
if ~isequal(size(computed),size(reference))
    error('%s shape mismatch: backend %s, MATLAB %s.',name, ...
          mat2str(size(computed)),mat2str(size(reference)));
end
difference = computed(:)-reference(:);
l2 = norm(difference,2);
linf = norm(difference,inf);
relative = l2/max(norm(reference(:),2),eps);
scaledMaximum = max(abs(difference)./max(1,abs(reference(:))));
fprintf('  %-14s L2=%11.4e  Linf=%11.4e  relL2=%11.4e  maxScaled=%11.4e\n', ...
        name,l2,linf,relative,scaledMaximum);
end
