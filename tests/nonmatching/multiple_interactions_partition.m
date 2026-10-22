function results = multiple_interactions_partition
% Reviewer-owned partition measurements. Thresholds require maintainer approval.
root=getenv('EXASIM_ROOT'); assert(~isempty(root),'EXASIM_ROOT not set');
names={'multiple_interactions_2d','multiple_interactions_3d_hextet'};
results=struct([]);
for c=1:numel(names)
    if contains(names{c},'3d')
        layouts={[1 1 1 1],[1 2 2 1],[1 2 3 1]};
    else
        layouts={[1 1 1 1],[1 2 2 1]};
    end
    folder=fullfile(root,'examples','CoupledProblems','Poisson',names{c}); addpath(folder);
    cleanup=onCleanup(@()rmpath(folder)); %#ok<NASGU> reference=[];
    for k=1:numel(layouts)
        outdir=tempname('/tmp'); mkdir(outdir);
        run_multiple_interactions(2,2,layouts{k},outdir);
        manifest=jsondecode(fileread(fullfile(outdir,'phase5_manifest.json')));
        fid=fopen(fullfile(outdir,'phase5_solution.bin'),'r'); assert(fid>=0); u=fread(fid,inf,'double'); fclose(fid);
        if isempty(reference), reference=u; difference=0;
        else, assert(numel(u)==numel(reference)); difference=norm(u-reference)/max(1,norm(reference)); end
        fprintf('MEASURE partition case=%s ranks=[%s] relative=%.17e interface={%s}\n', ...
            names{c},sprintf('%d ',layouts{k}),difference,jsonencode(manifest.per_rank_interface_counts));
        results=[results struct('case',names{c},'ranks',layouts{k},'difference',difference)]; %#ok<AGROW>
    end
    clear run_multiple_interactions; clear cleanup;
end
end
