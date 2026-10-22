function result = multiple_interactions_two_domain_baseline(caseName,porder,N,outdir)
% Legacy two-domain baseline on slabs 1--2 with the identical local meshes.
root=getenv('EXASIM_ROOT'); assert(~isempty(root),'EXASIM_ROOT not set');
assert(nargin==4 && (ischar(outdir) || isstring(outdir)),'explicit outdir is required');
work=char(outdir); assert(startsWith(work,filesep),'outdir must be absolute');
assert(~startsWith(work,[root filesep]),'outdir must lie outside the repository');
if ~isfolder(work), mkdir(work); end
caseDir=fullfile(root,'examples','CoupledProblems','Poisson',caseName);
assert(isfolder(caseDir),'unknown Phase 5 case %s',caseName); addpath(caseDir);
cleanupPath=onCleanup(@()rmpath(caseDir)); %#ok<NASGU>
is3d=contains(caseName,'3d'); mixed=contains(caseName,'quadtri') || contains(caseName,'hextet');
if is3d, families=[1 1]; if mixed, families=[1 0]; end
else, families=[1 1]; if mixed, families=[1 0]; end
end
old=pwd; cleanup=onCleanup(@()cd(old)); cd(work); %#ok<NASGU>
requestedPrefix=getenv('EXASIM_PREFIX');
run(fullfile(root,'frontends','Matlab','exasim_setup.m'));
if ~isempty(requestedPrefix), setenv('EXASIM_PREFIX',requestedPrefix); end
pde=cell(2,1); mesh=cell(2,1); reference=cell(2,1);
for d=1:2
 [pde{d},mesh{d}]=initializeexasim(); pde{d}.model="ModelD";
 role=ternary(mod(d,2)==1,"continuity","flux"); dim=ternary(is3d,"3d","2d");
 pde{d}.modelfile="pdemodel_"+role+"_"+dim; pde{d}.modelnumber=d;
 pde{d}.Cxxpreprocessing=1; pde{d}.platform="cpu"; pde{d}.mpiprocs=1; pde{d}.hybrid=1;
 pde{d}.porder=porder; pde{d}.pgauss=2*porder; pde{d}.physicsparam=1; pde{d}.tau=1;
 pde{d}.NLiter=12; pde{d}.NLtol=1e-12; pde{d}.linearsolvertol=1e-12;
 pde{d}.matvectol=1e-12; pde{d}.GMRESrestart=250; pde{d}.linearsolveriter=600;
 pde{d}.saveResNorm=1; pde{d}.interfacetype=1; pde{d}.interfacefluxmap=1;
 pde{d}.datapath=string(fullfile(work,'data')); pde{d}.builddir=string(fullfile(work,'build')); pde{d}.buildpath=pde{d}.builddir;
 nt=N*(1+mod(d+1,2)); xmin=(d-1)/4; xmax=d/4;
 if is3d
  [mesh{d}.p,mesh{d}.t]=cubemesh(N,nt,nt,families(d)); mesh{d}.p(1,:)=xmin+0.25*mesh{d}.p(1,:);
  mesh{d}.boundaryexpr=boundaries3d(xmin,xmax);
  mesh{d}.boundarycondition=ternary(d==1,[1;3;1;1;1;1],[1;1;1;2;1;1]);
  mesh{d}.interfacecondition=ternary(d==1,[0;1;0;0;0;0],[0;0;0;1;0;0]);
 else
  [mesh{d}.p,mesh{d}.t]=squaremesh(N,nt,1,families(d)); mesh{d}.p(1,:)=xmin+0.25*mesh{d}.p(1,:);
  mesh{d}.boundaryexpr=boundaries2d(xmin,xmax);
  mesh{d}.boundarycondition=ternary(d==1,[1;3;1;1],[1;1;1;2]);
  mesh{d}.interfacecondition=ternary(d==1,[0;1;0;0],[0;0;0;1]);
 end
 expected=ternary(is3d,4+4*families(d),3+families(d)); assert(size(mesh{d}.t,1)==expected);
 f=facenumbering(mesh{d}.p,mesh{d}.t,families(d),mesh{d}.boundaryexpr,mesh{d}.periodicexpr);
 reference{d}=createdgnodes(mesh{d}.p,mesh{d}.t,f,mesh{d}.curvedboundary,mesh{d}.curvedboundaryexpr,porder);
end
[sol,pde,mesh,master,dmd]=exasim(pde,mesh);
errors=zeros(1,2);
for d=1:2
 if numel(dmd{d})==1
  ne=size(sol{d},3); ep=dmd{d}{1}.elempart(:); ordered=sol{d}; ordered(:,:,ep(1:ne))=sol{d}; sol{d}=ordered;
 end
 errors(d)=l2error(sol{d},reference{d},master{d},is3d);
 readhistory(work,d,pde{d}.NLtol);
end
result=struct('case',caseName,'p',porder,'N',N,'errors',errors,'combinedError',norm(errors),'workdir',work);
fprintf('MEASURE two-domain-baseline case=%s p=%d N=%d errors=[%.17e %.17e] combined=%.17e work=%s\n', ...
 caseName,porder,N,errors(1),errors(2),result.combinedError,work);
end

function value=ternary(condition,a,b), if condition, value=a; else, value=b; end, end
function expr=boundaries2d(xmin,xmax)
expr={@(p)abs(p(2,:))<1e-8,@(p)abs(p(1,:)-xmax)<1e-8,@(p)abs(p(2,:)-1)<1e-8,@(p)abs(p(1,:)-xmin)<1e-8};
end
function expr=boundaries3d(xmin,xmax)
expr={@(p)abs(p(2,:))<1e-8,@(p)abs(p(1,:)-xmax)<1e-8,@(p)abs(p(2,:)-1)<1e-8,@(p)abs(p(1,:)-xmin)<1e-8,@(p)abs(p(3,:))<1e-8,@(p)abs(p(3,:)-1)<1e-8};
end
function readhistory(work,domain,nltol)
path=fullfile(work,'data',['dataout' num2str(domain)],'out_residualnorms0.bin'); assert(isfile(path));
fid=fopen(path,'r'); values=fread(fid,inf,'double'); fclose(fid); assert(~isempty(values)&&mod(numel(values),4)==0);
history=reshape(values,4,[]).'; assert(all(isfinite(history),'all')); assert(history(end,1)<=nltol);
end
function err=l2error(u,xdg,master,is3d)
s=master.shapegt; phi=s(:,:,1); dr=s(:,:,2); ds=s(:,:,3); w=master.gwe(:); err=0;
for e=1:size(xdg,3)
 x=xdg(:,:,e); xg=phi*x; xr=dr*x; xs=ds*x;
 if is3d
  dt=s(:,:,4); xt=dt*x; jac=xr(:,1).*(xs(:,2).*xt(:,3)-xs(:,3).*xt(:,2))-xr(:,2).*(xs(:,1).*xt(:,3)-xs(:,3).*xt(:,1))+xr(:,3).*(xs(:,1).*xt(:,2)-xs(:,2).*xt(:,1));
  exact=sin(pi*xg(:,1)).*sin(pi*xg(:,2)).*sin(pi*xg(:,3));
 else
  jac=xr(:,1).*xs(:,2)-xr(:,2).*xs(:,1); exact=sin(pi*xg(:,1)).*sin(pi*xg(:,2));
 end
 err=err+sum(w.*abs(jac).*(phi*u(:,1,e)-exact).^2);
end
err=sqrt(err);
end
