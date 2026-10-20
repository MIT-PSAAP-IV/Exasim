function result=phase13_hex_baseline(porder,N)
%PHASE13_HEX_BASELINE Single-domain hex baseline on the A-M1 model-2 mesh.
here=fileparts(mfilename('fullpath')); repo=fileparts(fileparts(here));
work=fullfile(tempdir,sprintf('exasim_phase13_hex_baseline_p%d_N%d',porder,N));
if exist(work,'dir'),rmdir(work,'s');end; mkdir(work); old=pwd; cleanup=onCleanup(@()cd(old)); cd(work);
addpath(here); run(fullfile(repo,'frontends','Matlab','exasim_setup.m'));
prefix=getenv('EXASIM_PHASE9_PREFIX'); if isempty(prefix),prefix=getenv('EXASIM_PREFIX');end
assert(~isempty(prefix) && isfile(fullfile(prefix,'lib','cmake','Exasim','ExasimConfig.cmake')), ...
 'current installed Exasim prefix required; set EXASIM_PHASE9_PREFIX');
assert(isfile(fullfile(prefix,'include','backend','Discretization','nonmatchingrectangular.hpp')), ...
 'installed Exasim prefix predates Phase 13');
setenv('EXASIM_PREFIX',prefix); [pde,mesh]=initializeexasim();
pde.model="ModelD"; pde.modelfile="pdemodel_phase11_2";
pde.Cxxpreprocessing=1; pde.platform="cpu"; pde.mpiprocs=1; pde.hybrid=1;
pde.porder=porder; pde.pgauss=2*porder; pde.physicsparam=1; pde.tau=1;
pde.NLiter=12; pde.NLtol=1e-12; pde.linearsolvertol=1e-12; pde.matvectol=1e-12;
pde.GMRESrestart=250; pde.linearsolveriter=600;
pde.datapath=string(fullfile(work,'data')); pde.builddir=string(fullfile(work,'build')); pde.buildpath=pde.builddir;
[mesh.p,mesh.t]=cubemesh(N,2*N,2*N,1); mesh.p(1,:)=0.5+0.5*mesh.p(1,:);
mesh.boundaryexpr={@(p)abs(p(2,:))<1e-8,@(p)abs(p(1,:)-1)<1e-8,@(p)abs(p(2,:)-1)<1e-8, ...
 @(p)abs(p(1,:)-0.5)<1e-8,@(p)abs(p(3,:))<1e-8,@(p)abs(p(3,:)-1)<1e-8};
mesh.boundarycondition=ones(6,1); f=facenumbering(mesh.p,mesh.t,1,mesh.boundaryexpr,mesh.periodicexpr);
dgnodes=createdgnodes(mesh.p,mesh.t,f,mesh.curvedboundary,mesh.curvedboundaryexpr,porder);
logfile=fullfile(work,'phase13_hex_baseline_solver.log'); diary(logfile); diary on;
try
 [sol,pde,mesh,master]=exasim(pde,mesh); %#ok<ASGLU>
 diary off;
catch exception
 diary off;
 if isfile(logfile),fprintf(2,'%s',fileread(logfile));end
 rethrow(exception);
end
logtext=fileread(logfile); fprintf('%s',logtext);
finalResidual=linear_residual_phase13(logtext);
assert(finalResidual<1e-10,'hex baseline final residual %.3e exceeds 1e-10',finalResidual);
phi=master.shapegt(:,:,1); dr=master.shapegt(:,:,2); ds=master.shapegt(:,:,3); dt=master.shapegt(:,:,4);
w=master.gwe(:); err=0;
for e=1:size(dgnodes,3)
 X=dgnodes(:,:,e); xg=phi*X; xr=dr*X; xs=ds*X; xt=dt*X;
 jac=xr(:,1).*(xs(:,2).*xt(:,3)-xs(:,3).*xt(:,2))-xr(:,2).*(xs(:,1).*xt(:,3)-xs(:,3).*xt(:,1))+xr(:,3).*(xs(:,1).*xt(:,2)-xs(:,2).*xt(:,1));
 uh=phi*sol(:,1,e); ue=exp(xg(:,1)).*sin(pi*xg(:,2)).*sin(pi*xg(:,3)); err=err+sum(w.*abs(jac).*(uh-ue).^2);
end
result=struct('p',porder,'N',N,'error',sqrt(err),'finalResidual',finalResidual);
fprintf('PHASE13 HEX BASELINE p=%d N=%d error=%.9e residual=%.3e\n', ...
 porder,N,result.error,finalResidual);
end

function residual=linear_residual_phase13(logtext)
initial=regexp(logtext,'Newton Iteration:\s*0,\s*Residual Norm:\s*([-+0-9.eE]+)','tokens','once');
assert(~isempty(initial),'hex baseline initial Newton residual not found');
updates=regexp(logtext,['Newton Iteration:\s*[0-9]+,\s*Alpha:\s*[-+0-9.eE]+,\s*Original Norm:' ...
 '\s*[-+0-9.eE]+,\s*Updated Norm:\s*([-+0-9.eE]+)'],'tokens');
assert(~isempty(updates),'hex baseline accepted Newton residual not found');
residual=str2double(updates{end}{1});
end
