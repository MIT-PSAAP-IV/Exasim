function measurements = multiple_interactions_edge_ablation
% Diagnostic only: remove one edge and report converged-solution sensitivity.
% This is not the fixed-state residual/J*v ablation required for acceptance:
% re-solving can legitimately change non-endpoint domains through the chain.
root=getenv('EXASIM_ROOT'); assert(~isempty(root),'EXASIM_ROOT not set');
cases={'multiple_interactions_2d','multiple_interactions_3d_hextet'};
measurements=struct([]);
for c=1:numel(cases)
    name=cases{c};
    folder=fullfile(root,'examples','CoupledProblems','Poisson',name); addpath(folder);
    cleanup=onCleanup(@()rmpath(folder)); %#ok<NASGU>
    clear run_multiple_interactions;
    intactOutdir=tempname('/tmp'); mkdir(intactOutdir);
    run_multiple_interactions(1,2,[1 1 1 1],intactOutdir,[1 1 1]);
    intactManifest=jsondecode(fileread(fullfile(intactOutdir,'phase5_manifest.json')));
    u0=read_solution(fullfile(intactOutdir,'phase5_solution.bin'));
    offsets=cumsum([0 domain_solution_counts(intactManifest)]);
    assert(offsets(end)==numel(u0),'manifest and solution sizes disagree');
    for removed=1:3
        active=ones(1,3); active(removed)=0;
        reducedOutdir=tempname('/tmp'); mkdir(reducedOutdir);
        run_multiple_interactions(1,2,[1 1 1 1],reducedOutdir,active);
        u=read_solution(fullfile(reducedOutdir,'phase5_solution.bin'));
        assert(numel(u)==numel(u0)); perDomain=zeros(1,4);
        for d=1:4
            ids=offsets(d)+1:offsets(d+1); perDomain(d)=max(abs(u(ids)-u0(ids)));
        end
        endpoints=[removed removed+1]; nonendpoints=setdiff(1:4,endpoints);
        assert(all(perDomain(endpoints)>0),'removing edge %d did not change both endpoint solutions',removed);
        fprintf('DIAGNOSTIC converged-solution-ablation case=%s removed=%d endpoint=[%s] nonendpoint=[%s]\n', ...
            name,removed,sprintf('%.17e ',perDomain(endpoints)),sprintf('%.17e ',perDomain(nonendpoints)));
        measurements=[measurements struct('case',name,'removed',removed,'perDomain',perDomain)]; %#ok<AGROW>
    end
    clear run_multiple_interactions cleanup;
end
end
function u=read_solution(path)
fid=fopen(path,'r'); assert(fid>=0); u=fread(fid,inf,'double'); fclose(fid);
end
function counts=domain_solution_counts(manifest)
p=double(manifest.porder); dim=double(manifest.dimension);
families=double(manifest.families(:).'); elements=double(manifest.element_counts(:).');
assert(numel(families)==4 && numel(elements)==4);
counts=zeros(1,4);
for d=1:4
    if families(d)==1
        npe=(p+1)^dim;
    elseif dim==2
        npe=(p+1)*(p+2)/2;
    else
        npe=(p+1)*(p+2)*(p+3)/6;
    end
    counts(d)=npe*elements(d);
end
end
