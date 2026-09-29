% Backend AV continuation and mesh-adaptivity counterpart of pdeapp_frontend.m.
casepath = fileparts(mfilename('fullpath'));
run(fullfile(casepath, '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

[pde,~] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel";
pde.platform = "cpu";
pde.mpiprocs = 8;
pde.hybrid = 1;
pde.porder = 2;
pde.saveParaview = 1;

mesh = mkmesh_auplate2d4UR(pde.porder);
%% ---- Boundary conditions ------------------------------------------------
% symmetry, inflow, iso-thermal wall, iso-thermal wall, outflow, inflow
mesh.boundarycondition = [4 1 3 3 2 1];

gam = 1.4; Re = 8.9e6; Pr = 0.71; Minf = 5.38;
Tref = 1020; Twall = 296;
pinf = 1/(gam*Minf^2); Tinf = pinf/(gam-1);
angleOfAttack = 0; rinf = 1.0;
ruinf = cos(angleOfAttack); rvinf = sin(angleOfAttack);
rEinf = 0.5+pinf/(gam-1);

pde.tau = 10.0;
pde.GMRESrestart = 250;
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6;
pde.linearsolveriter = 500;
pde.RBdim = 0;
pde.ppdegree = 0;
pde.NLtol = 1e-6;
pde.NLiter = 5;
pde.matvectol = 1e-6;

pde.AV = 1;
pde.AVcontinuationIter = 10;
pde.AVcontinuationLogScale = 2.0;
pde.AVcoeffStart = 0.02;
pde.AVcoeffEnd = 1e-5;
pde.AVdistfunction = 1;
pde.distanceboundaryconditions = [3]; % boundary-condition IDs stored in backend mesh.bf
pde.AVsmoothingMethod = 1;
pde.AVHelmholtzCoeff = 0.0001;
AVmaxdiv = 4.0; AVdistcoeff = 1.5e3;
pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref ...
    Twall AVmaxdiv AVdistcoeff pde.AVcoeffStart pde.AVcoeffEnd];


pde.avparam1 = [0.05 0.01  0.004  0.002 0.001 0.0005 0.00025 0.00012 0.00006];
pde.avparam2 = 0*pde.avparam1;
pde.meshadaptenabled = 0;

mesh.f = facenumbering(mesh.p,mesh.t,pde.elemtype,mesh.boundaryexpr,mesh.periodicexpr);
dist = meshdist3(mesh.f,mesh.dgnodes,mesh.perm,[3 4]);
mesh.vdg = zeros(size(mesh.dgnodes,1),2,size(mesh.dgnodes,3));
ld = 1./(1 + 1e3*mesh.dgnodes(:,1,:).^2);
mesh.vdg(:,1,:) = dist.*ld;

ui = [rinf ruinf rvinf rEinf];
UDG = initu(mesh,{ui(1),ui(2),ui(3),ui(4)});
UDG(:,2,:) = UDG(:,2,:).*tanh(100*dist);
UDG(:,3,:) = UDG(:,3,:).*tanh(100*dist);
TnearWall = Tinf * (Twall/Tref-1) * exp(-100*dist) + Tinf;
UDG(:,4,:) = TnearWall + 0.5*(UDG(:,2,:).^2 + UDG(:,3,:).^2);
mesh.udg = UDG;

% An export-only caller can generate this exact configured case without solving.
if exist('text2code_export_directory', 'var') && ~isempty(text2code_export_directory)
    exporttext2code(pde,mesh,text2code_export_directory);
    if exist('text2code_export_only', 'var') && text2code_export_only
        return;
    end
end

pde.gencode=1;
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);

mesh.udg = sol;
pde.dt = [0.01 0.1 1];
pde.saveSolFreq = length(pde.dt);
pde.avparam1 = [0.00005 0.00002 0.00001 0.000006];
pde.avparam2 = [1e-5 2e-5 3e-5 5e-5];
[sol,pde,mesh,master,dmd] = exasim(pde,mesh);

vdg = getsolution('dataout/outvdg',dmd, master.npe);

figure(1); clf; scaplot(mesh, vdg(:,1,:),[],2);
axis equal; axis tight; colorbar;

figure(2); clf; scaplot(mesh, real(eulereval(sol, 'M',gam,Minf)),[0 Minf],2);
axis equal; axis tight; colorbar; colormap('jet');

figure(3); clf; scaplot(mesh, eulereval(sol, 'r',gam,Minf),[],1); colorbar;

figure(4); clf; scaplot(mesh, eulereval(sol, 'u',gam,Minf),[0 1],2); colorbar;

mu = pde.physicsparam;
avField = (pde.avparam1(end) + pde.avparam2(end)*vdg(:,2,:)).*tanh(mu(end-2)*vdg(:,1,:));
figure(1); clf; scaplot(mesh, avField(:,1,:),[],2);
axis equal; axis tight; colorbar;
