% Add Exasim to Matlab search path
cdir = pwd(); ii = strfind(cdir, "Exasim");
run(cdir(1:(ii+5)) + "/install/setpath.m");

% initialize pdehm structure and mesh structure
[pdehm,~] = initializeexasim();

% pdehm.buildpath = string(pwd()) + "/hm";
% if exist(pdehm.buildpath, 'dir') == 0
%     mkdir(pdehm.buildpath);
% end
pdehm.datapath = string(pwd()) + "/hm";
pdehm.builddir = string(pwd()) + "/hm";
pdehm.buildpath = string(pwd()) + "/hm";

hm_path = pdehm.buildpath;
if exist(hm_path,'dir') == 0
    mkdir(hm_path);
end

% Define a pdehm model: governing equations, initial solutions, and boundary conditions
pdehm.model = "ModelD";          % ModelC, ModelD, ModelW
pdehm.modelfile = "pdemodel_hm";    % name of a file defining the pdehm model

% Choose computing platform and set nuomber of processors
pdehm.platform = "cpu";         % choose this option if NVIDIA GPUs are available
pdehm.mpiprocs = 1;             % number of MPI processors
pdehm.hybrid = 1;               % 0 -> LDG, 1 -> HDG
pdehm.debugmode = 0;
pdehm.nd = 2;

% Set discretization parameters, physical parameters, and solver parameters
pdehm.porder = 2;             % polynomial degree
pdehm.pgauss = 2*pdehm.porder;
pdehm.physicsparam = 0.01;       
pdehm.tau = 1.0;              % DG stabilization parameter
pde.GMRESrestart = 250;         %try 50
pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6; % GMRES tolerance
pde.linearsolveriter = 500; %try 100
pde.preconditioner = 1;
pdehm.ppdegree = 1;          % degree of polynomial preconditioner
pdehm.RBdim = 0;

meshhm = mkmesh_auplate2d(pdehm.porder);
meshhm.boundarycondition = [1;1;1;1;1]; 

[~,cgelcon,rowent2elem,colent2elem,~] = mkcgent2dgent(meshhm.dgnodes,1e-8);

div = limiting(divergence(sol, mesh.dgnodes, 1),0,divmax,1e3,0);
div = div/max(div(:)); 
div = limiting(div,0,1,1e3,0.1);
div = div/max(div(:));
meshhm.vdg = div;

% meshhm.vdg = dg2cg2(meshhm.vdg, cgelcon, colent2elem, rowent2elem);

[~, ~, jac] = volgeom(master.shapent,permute(meshhm.dgnodes,[1 3 2]));
jac = reshape(jac,[],1,size(meshhm.dgnodes,3));
jac = dg2cg2(jac, cgelcon, colent2elem, rowent2elem);
jac = dg2cg2(jac, cgelcon, colent2elem, rowent2elem);
meshhm.vdg(:,2,:) = jac.^0.5;%.^.25;%.^0.5;

% call exasim to generate and run C++ code to solve the pdehm model
pdehm.gencode = 1;
[solhm,pdehm,meshhm] = exasim(pdehm,meshhm);
s = solhm(:,1,:);
s = s/max(s(:));
s = limiting(s,0.0,0.5,1e3,0.0);
s = s/max(s(:));

S0=0.05;%S0 = 0.05; 
gamma = 1e3;
av = (s-S0).*(atan(gamma*(s-S0))/pi + 0.5) - atan(gamma)/pi + 0.5;    
av = av.*tanh(1e3*dist.*ld);

% plot solution
figure(1); clf; scaplot(meshhm,div(:,1,:)); axis on; axis equal;

figure(2); clf; scaplot(meshhm,solhm(:,1,:)); axis on; axis equal;

%figure(3); clf; scaplot(meshhm,meshhm.vdg(:,1,:)); axis on; axis equal;

figure(4); clf; scaplot(meshhm,av(:,1,:)); axis on; axis equal; colorbar;

%figure(5); clf; scaplot(meshhm,s(:,1,:)); axis on; axis equal;

