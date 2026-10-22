function multiple_interactions_frontends(root, output)
% Phase 1 MATLAB validator and golden-table driver (no code generation or solve).
addpath(fullfile(root,'frontends','Matlab','Preprocessing'));
addpath(fileparts(output));
controls = struct('temporalscheme',0,'torder',1,'nstage',1,'dt',0, ...
    'NLiter',20,'NLtol',1e-8,'linearsolver',0,'linearsolvertol',1e-8, ...
    'GMRESrestart',25,'GMRESortho',0,'linearsolveriter',200,'preconditioner',0);
pde = cell(3,1); mesh = cell(3,1);
p = [0 1 1 0;0 0 1 1]; t = [1;2;3;4];
expr = {@(x) abs(x(2,:))<1e-12,@(x) abs(x(1,:)-1)<1e-12, ...
        @(x) abs(x(2,:)-1)<1e-12,@(x) abs(x(1,:))<1e-12};
for d=1:3
    pde{d}=controls; pde{d}.ncu=1; pde{d}.elemtype=1; pde{d}.modelfile='phase1_model';
    mesh{d}=struct('p',p,'t',t,'boundaryexpr',{expr},'periodicexpr',[], ...
        'periodicboundary',[],'boundarycondition',[1 2 3 4],'interfacecondition',[]);
end
interaction(1)=struct('domains',[1 2],'type',1,'interface1',2,'condition1',1, ...
    'fluxmap1',1,'interface2',4,'condition2',1,'fluxmap2',1);
interaction(2)=struct('domains',[2 3],'type',1,'interface1',2,'condition1',1, ...
    'fluxmap1',1,'interface2',4,'condition2',1,'fluxmap2',1);
golden=[1 3 2 11 0 1 1 2 1 1 0 4 1 1 0 11 1 2 1 2 1 1 0 4 1 1 0]';
[table,~,~,~]=interactiongraph(pde,mesh,interaction);
assert(isequal(table,golden)); fid=fopen(output,'w'); fwrite(fid,table,'double'); fclose(fid);

% The validator must derive simplex/tensor-product type from mesh topology,
% not from this deliberately stale user-level elemtype value.
trimesh=mesh; trip=[p [0.5;0.5]]; trit=[1 2 3 4;2 3 4 1;5 5 5 5];
for d=1:3, trimesh{d}.p=trip; trimesh{d}.t=trit; end
assert(isequal(interactiongraph(pde,trimesh,interaction),golden));
tetmesh=mesh; tetp=[0 1 0 0;0 0 1 0;0 0 0 1]; tett=[1;2;3;4];
tetexpr={@(x) abs(x(1,:))<1e-12,@(x) abs(x(2,:))<1e-12, ...
         @(x) abs(x(3,:))<1e-12,@(x) abs(sum(x,1)-1)<1e-12};
for d=1:3
    tetmesh{d}.p=tetp; tetmesh{d}.t=tett; tetmesh{d}.boundaryexpr=tetexpr;
end
assert(isequal(interactiongraph(pde,tetmesh,interaction),golden));

bad=rmfield(interaction,'condition1'); expectReject(@() interactiongraph(pde,mesh,bad),'rule1','exactly the eight');
bad=interaction; bad(1).domains=[1 1]; expectReject(@() interactiongraph(pde,mesh,bad),'rule2','domains is invalid');
bad=interaction; bad(1).type=0; expectReject(@() interactiongraph(pde,mesh,bad),'rule3','only type=1');
badmesh=mesh; badmesh{1}.boundaryexpr{5}=@(x) false(1,size(x,2)); badmesh{1}.boundarycondition=1:5; bad=interaction; bad(1).interface1=5; expectReject(@() interactiongraph(pde,badmesh,bad),'rule4','selects no boundary face');
badmesh=mesh; badmesh{1}.boundarycondition=[1 2 3 2]; expectReject(@() interactiongraph(pde,badmesh,interaction),'rule5','nonunique boundary condition');
bad=interaction; bad(2).interface1=4; expectReject(@() interactiongraph(pde,mesh,bad),'rule6','overlaps another interaction');
bad=interaction; bad(1).condition1=0; expectReject(@() interactiongraph(pde,mesh,bad),'rule7','positive integer');
bad=interaction; bad(1).fluxmap1=[]; expectReject(@() interactiongraph(pde,mesh,bad),'rule8-empty','fluxmap1 is invalid');
bad=interaction; bad(1).fluxmap1=2; expectReject(@() interactiongraph(pde,mesh,bad),'rule8-range','fluxmap1 is invalid');
badpde=pde; badpde{1}.modelfile='phase1_no_fint'; expectReject(@() interactiongraph(badpde,mesh,interaction),'rule9','does not define fint');
badmesh=mesh; badmesh{1}.interfacecondition=[0 1 0 0]; expectReject(@() interactiongraph(pde,badmesh,interaction),'rule10','cannot be combined');
badpde=pde; badpde{2}.torder=2; expectReject(@() interactiongraph(badpde,mesh,interaction),'rule11-torder','solver control torder');
badpde=pde; badpde{2}.nstage=2; expectReject(@() interactiongraph(badpde,mesh,interaction),'rule11-nstage','solver control nstage');
badmesh=mesh; badmesh{1}.periodicboundary=2; expectReject(@() interactiongraph(pde,badmesh,interaction),'periodic','selects a periodic boundary');
badpde=pde; for d=1:3, badpde{d}.ncu=2; end
bad=interaction; bad(2).fluxmap2=[1 2];
expectReject(@() interactiongraph(badpde,mesh,bad),'width','one fint output width');

% Phase 2 selector-union partition contract. The one-selector vector is
% identical to the legacy scalar, and the union stores each element once.
pt=[1 2 3;2 3 4]; pt2t=zeros(2,3); pf=[1 2 1;2 3 4;3 4 2;4 1 3];
legacy=elementpartitionhdg(pt,pt2t,pf,2,1,0);
single=elementpartitionhdg(pt,pt2t,pf,[2],1,0);
unionpart=elementpartitionhdg(pt,pt2t,pf,[2 4],1,0);
empty=elementpartitionhdg(pt,pt2t,pf,[],1,0);
assert(isequal(single{1}.elempart,legacy{1}.elempart));
assert(isequal(single{1}.elempartpts,legacy{1}.elempartpts));
assert(isequal(unionpart{1}.elempart(:)',[1 2 3]));
assert(isequal(unionpart{1}.elempartpts,[0 3 0]));
assert(isequal(empty{1}.elempart,1:3));
fprintf('PASS: MATLAB interaction validator rules and golden table\n');
end

function expectReject(callback,label,expected)
try
    callback();
catch ME
    if contains(ME.message,expected), return; end
    error('validator rejected %s with the wrong diagnostic: %s',label,ME.message);
end
error('validator accepted rule mutation: %s',label);
end
