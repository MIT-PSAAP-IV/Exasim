function results = phase10_convergence(mode, orientationFilter, pairFilter, compositeFilter)
%PHASE10_CONVERGENCE Mixed-family manufactured Poisson convergence.
% ci: p=1,2, N=1,2,4, pair A, A1 on.
% full: p=1..4, N=1,2,4,8, pairs A/S/R, A1 off/on.
% curved: p=1..4, N=1,2,4,8, independently interpolated curved interface.
if nargin < 1, mode = 'ci'; end
switch mode
    case 'ci'
        porders = 1:2; levels = [1 2 4]; pairs = {'A'}; compositeModes = true; curved = false;
    case 'full'
        porders = 1:4; levels = [1 2 4 8]; pairs = {'A','S','R'}; compositeModes = [false true]; curved = false;
    case 'curved'
        porders = 1:4; levels = [1 2 4 8]; pairs = {'A'}; compositeModes = false; curved = true;
    otherwise
        error('unknown mode %s', mode);
end
orientations = {'tri-quad', 'quad-tri'};
if nargin >= 2 && ~isempty(orientationFilter), orientations = {char(orientationFilter)}; end
if nargin >= 3 && ~isempty(pairFilter), pairs = {char(pairFilter)}; end
if nargin >= 4 && ~isempty(compositeFilter), compositeModes = logical(compositeFilter); end
results = struct('orientation', {}, 'pair', {}, 'composite', {}, 'curved', {}, ...
    'p', {}, 'N', {}, 'e1', {}, 'e2', {});
for orientation = orientations
    for pair = pairs
        for composite = compositeModes
            for p = porders
                for N = levels
                    out = phase10_fullsolve(orientation{1}, 1, 1, p, N, composite, pair{1}, curved);
                    e1 = physical_l2(out.sol{1}, out.mesh{1}.dgnodes, out.master{1});
                    e2 = physical_l2(out.sol{2}, out.mesh{2}.dgnodes, out.master{2});
                    results(end+1) = struct('orientation', orientation{1}, 'pair', pair{1}, ...
                        'composite', composite, 'curved', curved, 'p', p, 'N', N, ...
                        'e1', e1, 'e2', e2); %#ok<AGROW>
                    fprintf('PHASE10 conv %s %s A1=%d curved=%d p=%d N=%d e1=%.6e e2=%.6e\n', ...
                        orientation{1}, pair{1}, composite, curved, p, N, e1, e2);
                end
                rows = results(strcmp({results.orientation}, orientation{1}) & ...
                    strcmp({results.pair}, pair{1}) & [results.composite] == composite & ...
                    [results.curved] == curved & [results.p] == p);
                r1 = log2([rows(1:end-1).e1] ./ [rows(2:end).e1]);
                r2 = log2([rows(1:end-1).e2] ./ [rows(2:end).e2]);
                fprintf('PHASE10 conv %s %s A1=%d curved=%d p=%d rates model1', ...
                    orientation{1}, pair{1}, composite, curved, p); fprintf(' %.3f', r1);
                fprintf(' model2'); fprintf(' %.3f', r2); fprintf('\n');
                if composite
                    target = p + 1 - 0.2;
                    assert(r1(end) >= target && r2(end) >= target, ...
                        'Phase 10 %s %s p=%d A1 rates miss predeclared %.1f threshold', ...
                        orientation{1}, pair{1}, p, target);
                elseif curved && p <= 2
                    target = p + 1 - 0.3;
                    assert(r1(end) >= target && r2(end) >= target, ...
                        'Phase 10 curved %s p=%d rates miss Phase 9 %.1f threshold', ...
                        orientation{1}, p, target);
                end
            end
        end
    end
end
fprintf('PASS phase10_convergence %s\n', mode);
end

function err = physical_l2(u, dg, master)
phi = master.shapegt(:,:,1);
d0 = master.shapegt(:,:,2);
d1 = master.shapegt(:,:,3);
err = 0;
for e = 1:size(dg,3)
    X = dg(:,:,e);
    xg = phi*X;
    jac = (d0*X(:,1)).*(d1*X(:,2)) - (d1*X(:,1)).*(d0*X(:,2));
    uh = phi*u(:,1,e);
    ue = sin(pi*xg(:,1)).*sin(pi*xg(:,2));
    err = err + sum(master.gwe.*abs(jac).*(uh-ue).^2);
end
err = sqrt(err);
end
