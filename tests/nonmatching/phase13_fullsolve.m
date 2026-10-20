function result = phase13_fullsolve(ordering, sequence, porder, N, k1, k2, nonlinear)
%PHASE13_FULLSOLVE Manual CPU mixed tet/hex solve; not registered in CTest.
if nargin<1, ordering='A'; end
if nargin<2, sequence=ordering+"-M1"; end
if nargin<3, porder=2; end
if nargin<4, N=2; end
if nargin<5, k1=1; end
if nargin<6, k2=1; end
if nargin<7, nonlinear=false; end
ordering=char(ordering); sequence=char(sequence);
assert(ismember(ordering,{'A','B'}));
assert(ismember(sequence,{'A-M1','A-M1-fine','A-M2','B-M2'}));
assert(porder>=1 && porder<=4 && N>=1 && k1>=1 && k2>=1);
if ordering=='A', elem=[0 1]; else, elem=[1 0]; end
switch sequence
 case 'A-M1',      cells=[N 2*N 2*N; N 2*N 2*N]; nested=true;
 case 'A-M1-fine', cells=[N 2*N 2*N; N N N];     nested=true;
 case 'A-M2',      cells=[N N N; N 2*N 2*N];     nested=false;
 case 'B-M2',      cells=[N 2*N 2*N; N 2*N 2*N]; nested=false;
end
assert((ordering=='A')==startsWith(sequence,'A-'),'ordering and sequence disagree');

here=fileparts(mfilename('fullpath')); repo=fileparts(fileparts(here));
work=fullfile(tempdir,sprintf('exasim_phase13_%s_%s_p%d_N%d_%d_%d',ordering,sequence,porder,N,k1,k2));
if exist(work,'dir'), rmdir(work,'s'); end
mkdir(work); old=pwd; cleanup=onCleanup(@()cd(old)); cd(work); addpath(here);
run(fullfile(repo,'frontends','Matlab','exasim_setup.m'));
prefix=getenv('EXASIM_PHASE9_PREFIX');
if isempty(prefix), prefix=getenv('EXASIM_PREFIX'); end
assert(~isempty(prefix) && isfile(fullfile(prefix,'lib','cmake','Exasim','ExasimConfig.cmake')), ...
 'current installed Exasim prefix required; set EXASIM_PHASE9_PREFIX');
assert(isfile(fullfile(prefix,'include','backend','Discretization','nonmatchingrectangular.hpp')), ...
 'installed Exasim prefix predates Phase 13');
setenv('EXASIM_PREFIX',prefix);

pde=cell(2,1); mesh=cell(2,1); ranks=[k1 k2]; referenceDgnodes=cell(2,1);
for model=1:2
 [pde{model},mesh{model}]=initializeexasim();
 pde{model}.model="ModelD";
 if nonlinear, pde{model}.modelfile="pdemodel_phase13_nl_"+string(model);
 else, pde{model}.modelfile="pdemodel_phase11_"+string(model); end
 pde{model}.modelnumber=model; pde{model}.Cxxpreprocessing=1; pde{model}.platform="cpu";
 pde{model}.mpiprocs=ranks(model); pde{model}.hybrid=1; pde{model}.porder=porder;
 pde{model}.pgauss=2*porder; pde{model}.physicsparam=1; pde{model}.tau=1;
 pde{model}.NLiter=12; pde{model}.NLtol=1e-12; pde{model}.linearsolvertol=1e-12;
 pde{model}.matvectol=1e-12; pde{model}.GMRESrestart=250; pde{model}.linearsolveriter=600;
 pde{model}.interfacetype=1; pde{model}.interfacefluxmap=1;
 pde{model}.datapath=string(fullfile(work,'data')); pde{model}.builddir=string(fullfile(work,'build'));
 pde{model}.buildpath=pde{model}.builddir;
 [mesh{model}.p,mesh{model}.t]=cubemesh(cells(model,1),cells(model,2),cells(model,3),elem(model));
 if model==1, mesh{model}.p(1,:)=0.5*mesh{model}.p(1,:);
 else, mesh{model}.p(1,:)=0.5+0.5*mesh{model}.p(1,:); end
 mesh{model}.boundaryexpr=cube_boundaries_phase13(0.5*(model-1),0.5*model);
 if model==1
  mesh{model}.boundarycondition=[1;2;1;1;1;1]; mesh{model}.interfacecondition=[0;1;0;0;0;0];
 else
  mesh{model}.boundarycondition=[1;1;1;2;1;1]; mesh{model}.interfacecondition=[0;0;0;1;0;0];
 end
 f=facenumbering(mesh{model}.p,mesh{model}.t,elem(model),mesh{model}.boundaryexpr,mesh{model}.periodicexpr);
 referenceDgnodes{model}=createdgnodes(mesh{model}.p,mesh{model}.t,f,mesh{model}.curvedboundary,mesh{model}.curvedboundaryexpr,porder);
end
interfaceMeasure=(max(mesh{1}.p(2,:))-min(mesh{1}.p(2,:)))* ...
 (max(mesh{1}.p(3,:))-min(mesh{1}.p(3,:)));
assert(interfaceMeasure>0,'Phase 13 interface measure must be positive');
interfaceFaceSize=sqrt(interfaceMeasure);
nestingTolerance=1000*eps(1.0)*max(1,interfaceFaceSize);
if nested
 fprintf('PHASE13 nesting tolerance %.9e = 1000*eps*max(1,faceSize), faceSize %.9e\n', ...
  nestingTolerance,interfaceFaceSize);
end
logfile=fullfile(work,'phase13_solver.log');
diary(logfile); diary on;
try
 [sol,pde,mesh,master,dmd]=exasim(pde,mesh);
 diary off;
catch exception
 diary off;
 if isfile(logfile), fprintf(2,'%s',fileread(logfile)); end
 rethrow(exception);
end
logtext=fileread(logfile); fprintf('%s',logtext);
assert(contains(logtext,'Non-matching donor search, Gamma^1') && contains(logtext,'Non-matching donor search, Gamma^2'));
interfaceElements=cell(2,1);
for model=1:2
 assert(all(cellfun(@(part)isfield(part,'intepartpts') && numel(part.intepartpts)>=2,dmd{model})), ...
  'coupled-interface partition metadata is missing for model %d',model);
 interfaceElements{model}=cellfun(@(part)part.intepartpts(2),dmd{model});
 assert(numel(interfaceElements{model})==ranks(model),'rank/interface-count mismatch for model %d',model);
 countsText=strtrim(sprintf('%d ',interfaceElements{model}));
 fprintf('PHASE13 interface elements model %d ranks=[%s]\n',model,countsText);
 if numel(dmd{model})==1
  ne=size(sol{model},3); ep=dmd{model}{1}.elempart(:);
  assert(numel(ep)>=ne && isequal(sort(ep(1:ne)).',1:ne), ...
   'single-rank elempart is not a permutation for model %d',model);
  u=sol{model}; u(:,:,ep(1:ne))=sol{model}; sol{model}=u;
 end
 mesh{model}.dgnodes=referenceDgnodes{model};
end
[e1,n1]=l2error_phase13(sol{1},mesh{1}.dgnodes,master{1});
[e2,n2]=l2error_phase13(sol{2},mesh{2}.dgnodes,master{2});
searchTokens=regexp(logtext,['Non-matching donor search, Gamma\^([12]) [^\r\n]*' ...
 'max distance = ([-+0-9.eE]+), points off donor face = ([0-9]+)'],'tokens');
distances=NaN(1,2); offFacePoints=NaN(1,2);
for token=searchTokens
 direction=str2double(token{1}{1}); distances(direction)=str2double(token{1}{2});
 offFacePoints(direction)=str2double(token{1}{3});
end
if nested || nonlinear
 assert(all(~isnan(distances)) && all(~isnan(offFacePoints)), ...
  'donor-distance/off-face diagnostics are missing');
end
maxdist=max(distances,[],'omitnan');
if nested
 assert(distances(1)<=nestingTolerance, ...
  'nested flux-direction donor distance %.3e exceeds tolerance %.3e',distances(1),nestingTolerance);
 assert(offFacePoints(1)==0,'nested flux direction has %d off-face points',offFacePoints(1));
end
if nonlinear
 assert(maxdist<=1e-14,'nonlinear donor distance %.3e exceeds 1e-14',maxdist);
end
nestingTokens=regexp(logtext,'mixed nesting violations, Gamma\^([12]) = ([0-9]+)','tokens');
nestingViolations=NaN(1,2);
for token=nestingTokens
 nestingViolations(str2double(token{1}{1}))=str2double(token{1}{2});
end
if nested
 assert(all(~isnan(nestingViolations)),'mixed nesting diagnostics are missing');
 assert(nestingViolations(1)==0, ...
  'flux-returning tet-owner direction selects multiple donor faces on %d owner faces',nestingViolations(1));
end
history=parse_history(logtext); finalResidual=history.residuals(end); ratios=[];
assert(finalResidual<1e-10,'final residual %.3e exceeds 1e-10',finalResidual);
if nonlinear
 residuals=history.residuals;
 assert(numel(residuals)>=3,'nonlinear solve produced fewer than three Newton residuals');
 mask=residuals(1:end-1)>1e-9; ratios=residuals(2:end)./residuals(1:end-1).^2;
 assert(sum(ratios(mask)<0.5)>=2,'nonlinear Newton quadratic-rate gate failed');
end
result=struct('ordering',ordering,'sequence',sequence,'p',porder,'N',N,'ranks',[k1 k2], ...
 'e1',e1,'e2',e2,'E',hypot(e1,e2),'ndof',n1+n2,'maxdist',maxdist, ...
 'distances',distances,'offFacePoints',offFacePoints,'nested',nested, ...
 'nestingViolations',nestingViolations,'interfaceElements',{interfaceElements}, ...
 'interfaceFaceSize',interfaceFaceSize,'nestingTolerance',nestingTolerance, ...
 'finalResidual',finalResidual,'ratios',ratios,'sol',{sol},'mesh',{mesh},'log',logtext);
fprintf('PHASE13 %s %s p=%d N=%d %d+%d e1=%.9e e2=%.9e E=%.9e maxdist=%.3e residual=%.3e\n', ...
 ordering,sequence,porder,N,k1,k2,e1,e2,result.E,maxdist,finalResidual);
end

function [err,ndof]=l2error_phase13(u,dgnodes,master)
phi=master.shapegt(:,:,1); dr=master.shapegt(:,:,2); ds=master.shapegt(:,:,3); dt=master.shapegt(:,:,4);
w=master.gwe(:); [npe,~,ne]=size(dgnodes); ndof=npe*ne; err=0;
for e=1:ne
 X=dgnodes(:,:,e); xg=phi*X; xr=dr*X; xs=ds*X; xt=dt*X;
 jac=xr(:,1).*(xs(:,2).*xt(:,3)-xs(:,3).*xt(:,2))-xr(:,2).*(xs(:,1).*xt(:,3)-xs(:,3).*xt(:,1))+xr(:,3).*(xs(:,1).*xt(:,2)-xs(:,2).*xt(:,1));
 uh=phi*u(:,1,e); ue=exp(xg(:,1)).*sin(pi*xg(:,2)).*sin(pi*xg(:,3));
 err=err+sum(w.*abs(jac).*(uh-ue).^2);
end
err=sqrt(err);
end
function expr=cube_boundaries_phase13(xmin,xmax)
expr={@(p)abs(p(2,:))<1e-8,@(p)abs(p(1,:)-xmax)<1e-8,@(p)abs(p(2,:)-1)<1e-8, ...
 @(p)abs(p(1,:)-xmin)<1e-8,@(p)abs(p(3,:))<1e-8,@(p)abs(p(3,:)-1)<1e-8};
end

function h = parse_history(logtext)
% Initial residual, then one accepted step per Newton iteration: the solver prints a
% line per backtracking trial and a final line with the accepted alpha and norm.
initial = regexp(logtext, 'Newton Iteration:\s*0,\s*Residual Norm:\s*([-+0-9.eE]+)', 'tokens', 'once');
assert(~isempty(initial), 'initial Newton residual not found');
lines = regexp(logtext, ['Newton Iteration:\s*([0-9]+),\s*Alpha:\s*([-+0-9.eE]+),\s*Original Norm:' ...
    '\s*([-+0-9.eE]+),\s*Updated Norm:\s*([-+0-9.eE]+)'], 'tokens');
it = cellfun(@(t) str2double(t{1}), lines);
[uniqueIt, last] = unique(it, 'last');
h.iterations = [0 uniqueIt(:).'];
h.residuals = [str2double(initial{1}) cellfun(@(t) str2double(t{4}), lines(last))];
h.alphas = cellfun(@(t) str2double(t{2}), lines(last));
h.backtracks = numel(lines) - numel(last);
gm = regexp(logtext, 'GMRES\(([0-9]+)\) converges[^\r\n]*?within\s*([0-9]+) iterations', 'tokens');
h.gmres = cellfun(@(t) str2double(t{2}), gm);
gmLines = regexp(logtext, 'GMRES\([0-9]+\) converges[^\r\n]*', 'match');
h.gmresLines = gmLines;
assert(isequal(h.iterations, 0:numel(h.residuals)-1), 'Newton iteration sequence is incomplete');
end
