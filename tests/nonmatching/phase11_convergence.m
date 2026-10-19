function results = phase11_convergence(configuration)
%PHASE11_CONVERGENCE M1 acceptance and M2/M3 diagnostic sweeps.
% Fixed acceptance: M1 last-level rates >= p+1-0.2. M2/M3 rates are reported.
if nargin < 1, configuration = 'M1'; end
assert(ismember(configuration, {'M1','M2','M3','M3o','M3c'}), 'unknown Phase 11 family');
results = struct('family',{},'configuration',{},'p',{},'N',{},'e1',{},'e2',{},'e',{});
for elem = {'tet','hex'}
    for p = 1:4
        rows = [];
        levels = [1 2 4 8];
        if startsWith(configuration,'M3') && strcmp(elem{1},'hex') && p == 1
            % One hex p=1 element per model hits a pre-existing base-solver
            % divergence (single-domain too; maintainer decision 2026-10-15).
            levels = [2 4 8 16];
        end
        for N = levels
            refMaxu = [];
            if startsWith(configuration,'M3')
                ref = phase11_fullsolve(elem{1},'M1',p,N,1); refMaxu = ref.maxu;
            end
            r = phase11_fullsolve(elem{1},configuration,p,N,1,refMaxu);
            row = struct('family',elem{1},'configuration',configuration,'p',p,'N',N, ...
                'e1',r.e1,'e2',r.e2,'e',r.e);
            results(end+1)=row; %#ok<AGROW>
            rows = [rows r.e]; %#ok<AGROW>
        end
        rates = log2(rows(1:end-1)./rows(2:end));
        fprintf('PHASE11 rates %s %s p=%d: %.6f %.6f\n',elem{1},configuration,p,rates);
        if strcmp(configuration,'M1')
            assert(rates(end)>=p+1-0.2, ...
                'M1 %s p=%d last rate %.6f below %.6f',elem{1},p,rates(end),p+1-0.2);
        end
    end
end
end
