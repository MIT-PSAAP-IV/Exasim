% Poisson on the unit cube: one HDG rank and one LDG rank.
caseDir=fileparts(mfilename('fullpath'));
if exist('initializeexasim','file')~=2
    setup=fullfile(caseDir,'..','..','..','frontends','Matlab','exasim_setup.m');
    if ~isfile(setup)
        error('Run your installed share/exasim/matlab/exasim_setup.m first.');
    end
    run(setup);
end
addpath(caseDir);
[pde,mesh]=initializeexasim();
pde.model="ModelD";
pde.modelfile="pdemodel";
pde.platform="cpu";
pde.mpiprocs=2;
nhdg=1;
pde.ncu=1;
pde.porder=2;
pde.pgauss=2*pde.porder;
pde.physicsparam=[1,0];
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
[mesh.p,mesh.t]=cubemesh(N,N,N,1);
mesh.boundaryexpr={@(p) p(2,:)<1e-8, @(p) p(1,:)>1-1e-8, ...
    @(p) p(2,:)>1-1e-8, @(p) p(1,:)<1e-8, ...
    @(p) p(3,:)<1e-8, @(p) p(3,:)>1-1e-8};
mesh.boundarycondition=ones(6,1);
labels=mean(reshape(mesh.p(1,mesh.t),size(mesh.t)),1)<=0.5;
if ~isempty(getenv('HDGLDG_RUN_DIR'))
    pde.datapath=string(getenv('HDGLDG_RUN_DIR'));
    pde.builddir=fullfile(pde.datapath,'.exasim');
end
[sol,pde,mesh]=hdgldgsolve(pde,mesh,labels,nhdg);
mesh.porder=pde.porder;
[mesh.dgnodes,mesh.elemtype]=createdgnodes(mesh.p,mesh.t,[],[],[],pde.porder);
[mesh.xpe,mesh.telem]=masternodes(pde.porder,3,mesh.elemtype);
disp('Solution components are [u,qx,qy,qz], with q=-grad(u).');
