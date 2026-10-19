% Coupled axisymmetric Navier-Stokes/solid-heat-transfer calculation.
clear pde mesh master dmd sol;

% First converge the stand-alone flow problem. The coupled Newton solve is
% initialized with this solution, following cylindermach8_matching.
pdeapp1;
fluidPde = pde;
fluidMesh = mesh;
fluidMesh.udg = sol;
fluidMesh.vdg = getsolution('dataout/outvdg',dmd, master.npe);

pdeapp2;

pde = {fluidPde,pde2};
mesh = {fluidMesh,mesh2};

% The independently generated triangle and quadrilateral meshes use the
% type-1 interface implementation introduced for mixed element families.
for model = 1:2
    pde{model}.interfacetype = 1;
    pde{model}.modelnumber = model - 1;
    pde{model}.combinedmodel = true;
end

mesh{1}.boundarycondition = [4;2;1;7];
mesh{1}.interfacecondition = [0;0;0;1];
pde{1}.interfacefluxmap = 4;
pde{1}.ibs = 7;
pde{1}.distanceboundaryconditions = 7;
% The stand-alone precursor already computed the wall-distance field. The
% auxiliary distance and Helmholtz AV solves are not communicator-safe inside
% a two-model run. Keep AV active, but reuse that distance and do not smooth.
pde{1}.AVdistfunction = 0;
pde{1}.AVsmoothingMethod = 0;
pde{1}.AV = 0;
pde{1}.AVcontinuationIter = 0;
pde{1}.avparam1 = [];
pde{1}.avparam2 = [];
pde{1}.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall ...
                       AVmaxdiv AVdistcoeff 0 fluidPde.AVcoeffEnd];

% The monolithic solve inherits the precursor's nonlinear stopping rule.
pde{2}.NLiter = pde{1}.NLiter;
pde{2}.NLtol = pde{1}.NLtol;

mesh{2}.boundarycondition = [3;2;1];
mesh{2}.interfacecondition = [1;0;0];
pde{2}.interfacefluxmap = 1;

% Remove inactive rank files from older layouts/rank counts before writing the
% solid model. They are not read by the solver, but must not masquerade as
% inputs produced by this run during cross-frontend verification.
for staleDirectory = ["datain1" "datain2" "dataout1" "dataout2"]
    if exist(char(staleDirectory), 'dir')
        rmdir(char(staleDirectory), 's');
    end
end

figure(5); clf;
meshplot(mesh{1}); hold on; meshplot(mesh{2});
axis on; axis equal; axis tight;
title('Non-matching fluid/solid interface');

if exist('isoq2d_coupled_setup_only','var') && isoq2d_coupled_setup_only
    return;
end

[sol,pde,mesh,master,dmd] = exasim(pde,mesh);

figure(6); clf;
scaplot(mesh{1},eulereval(sol{1},'t',gam,Minf),[],1,2); hold on;
scaplot(mesh{2},sol{2}(:,1,:),[],1,2);
axis on; axis equal; axis tight; colorbar;
title('Coupled fluid and solid temperature');
