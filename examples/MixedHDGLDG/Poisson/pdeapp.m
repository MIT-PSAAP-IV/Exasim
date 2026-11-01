% Run this script as for examples/Poisson/poisson2d/pdeapp.m.
caseDir=fileparts(mfilename('fullpath'));
setup=fullfile(caseDir,'..','..','..','frontends','Matlab','exasim_setup.m');
if exist('initializeexasim','file')~=2
    if isfile(setup)
        run(setup);
    else
        error('Run your installed share/exasim/matlab/exasim_setup.m first.');
    end
end
addpath(caseDir);

[pde,mesh]=initializeexasim();
pde.model="ModelD";
pde.modelfile="pdemodel";
pde.platform="cpu";
pde.mpiprocs=2;                 % total MPI ranks
nhdg=1;                        % first rank HDG, remaining rank LDG
pde.ncu=1;
pde.porder=2;
pde.pgauss=2*pde.porder;
pde.physicsparam=[1,0];         % conductivity, Dirichlet value
pde.tau=1;
pde.NLtol=1e-10;
pde.NLiter=20;
pde.linearsolvertol=1e-8;
pde.linearsolveriter=2000;
pde.GMRESrestart=200;
pde.GMRESortho=1;
pde.matvectol=1e-3;
pde.preconditioner=1;
pde.ppdegree=0;
pde.RBdim=0;

N=4;
[mesh.p,mesh.t]=squaremesh(N,N,1,1);
mesh.boundaryexpr={@(p) abs(p(2,:))<1e-8, @(p) abs(p(1,:)-1)<1e-8, ...
    @(p) abs(p(2,:)-1)<1e-8, @(p) abs(p(1,:))<1e-8};
mesh.boundarycondition=[1;1;1;1];
labels=mean(reshape(mesh.p(1,mesh.t),size(mesh.t)),1)<=0.5;

% Optional location override for automated checks; normal users need none.
if ~isempty(getenv('HDGLDG_RUN_DIR'))
    pde.datapath=string(getenv('HDGLDG_RUN_DIR'));
    pde.builddir=fullfile(pde.datapath,'.exasim');
end

[sol,pde,mesh]=hdgldgsolve(pde,mesh,labels,nhdg);
mesh.porder=pde.porder;
% The coupled preprocessor does not populate MATLAB's plotting metadata.
[mesh.dgnodes,mesh.elemtype]=createdgnodes(mesh.p,mesh.t,[],[],[],pde.porder);
[mesh.xpe,mesh.telem]=masternodes(pde.porder,size(mesh.p,1),mesh.elemtype);
if usejava('desktop')
    figure(1); clf; scaplot(mesh,sol(:,1,:),[],2); axis equal tight;
end
