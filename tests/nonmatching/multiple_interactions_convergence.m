function results = multiple_interactions_convergence
% Reviewer-owned pre-gate convergence measurements. No rate threshold is used.
root=getenv('EXASIM_ROOT'); assert(~isempty(root),'EXASIM_ROOT not set');
names={'multiple_interactions_2d','multiple_interactions_2d_quadtri', ...
       'multiple_interactions_3d','multiple_interactions_3d_hextet'};
results=struct([]);
for c=1:numel(names)
    folder=fullfile(root,'examples','CoupledProblems','Poisson',names{c}); addpath(folder);
    cleanup=onCleanup(@()rmpath(folder)); %#ok<NASGU>
    for p=1:4
        errors=zeros(1,3); baselines=zeros(1,3); levels=[2 4 8];
        for l=1:3
            outdir=tempname('/tmp'); mkdir(outdir);
            run_multiple_interactions(p,levels(l),[1 1 1 1],outdir);
            manifest=read_manifest(outdir); errors(l)=manifest.combined_l2;
            baselineOutdir=tempname('/tmp');
            baseline=multiple_interactions_two_domain_baseline(names{c},p,levels(l),baselineOutdir);
            baselines(l)=baseline.combinedError;
            results=[results struct('case',names{c},'p',p,'N',levels(l),'E',errors(l), ...
                'baseline',baselines(l),'workdir',outdir)]; %#ok<AGROW>
        end
        rates=log(errors(1:end-1)./errors(2:end))/log(2);
        baselineRates=log(baselines(1:end-1)./baselines(2:end))/log(2);
        fprintf('MEASURE convergence case=%s p=%d errors=[%s] rates=[%s] baseline=[%s] baseline_rates=[%s]\n', ...
            names{c},p,sprintf('%.17e ',errors),sprintf('%.17e ',rates), ...
            sprintf('%.17e ',baselines),sprintf('%.17e ',baselineRates));
    end
    clear run_multiple_interactions; clear cleanup;
end
end
function manifest=read_manifest(outdir)
path=fullfile(outdir,'phase5_manifest.json'); assert(isfile(path),'missing Phase 5 manifest');
manifest=jsondecode(fileread(path));
assert(isfinite(manifest.combined_l2),'non-finite combined L2 error');
end
