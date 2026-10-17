function results = phase9_convergence(mode)
%PHASE9_CONVERGENCE [nonmatching] Phase 9 manufactured-solution studies (C4/C5).
%
%   phase9_convergence('ci')         registered subset: straight unequal-h interface,
%                                    p = 1, 2, levels N = 1, 2, 4
%   phase9_convergence('straight')   complete sweep: p = 1, 2, 3, levels N = 1, 2, 4, 8
%   phase9_convergence('curved')     curved interface, p = 1, 2, 3, levels N = 1, 2, 4, 8
%   phase9_convergence('b1sweep')    B1 diagnostic: p = 1..4 on mesh pairs A, S, R
%   phase9_convergence('b1quad')     B1 diagnostic: face-quadrature sensitivity (A, R)
%   phase9_convergence('extension')  Gamma^1 extends 12.5% beyond Gamma^2, p = 2,
%                                    levels N = 1, 2, 4, 8, ranks 1+1 and 2+2
%
% Exact solution u = sin(pi x) sin(pi y) on both models (examples/CoupledProblems/Poisson/matching
% models: -lap u = 2 pi^2 u, Dirichlet data from u, continuity of u and flux).
% Errors are L2 norms with physical quadrature: tensor Gauss-Legendre with p+3 points
% per direction, the isoparametric map of the DG nodes and |det J| weights. The
% criteria below were fixed before the first solver run (see phase9_report.md).
%
% Model 1 (owner of Gamma^1): squaremesh(2N, 4N) on [0,0.5]x[0,1], h1 = 1/(4N).
% Model 2: squaremesh(3N, 6N) on [0.5,1]x[0,1], h2 = 1/(6N) (h ratio 3:2).

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
example = fullfile(repo, 'examples', 'CoupledProblems', 'Poisson', 'matching');
addpath(example); addpath(here);
work = fullfile(tempdir, ['exasim_phase9_conv_' mode]);
if exist(work, 'dir'), rmdir(work, 's'); end
mkdir(work);
old = pwd; cleanup = onCleanup(@() cd(old)); cd(work);
run(fullfile(repo, 'frontends', 'Matlab', 'exasim_setup.m'));
prefix = getenv('EXASIM_PHASE9_PREFIX');
if isempty(prefix), prefix = getenv('EXASIM_PHASE6_PREFIX'); end
assert(~isempty(prefix), 'Set EXASIM_PHASE9_PREFIX to the tested Exasim installation');
setenv('EXASIM_PREFIX', prefix);

switch mode
    case 'ci',        results = straight_study(work, 1:2, [1 2 4], false);
    case 'straight',  results = straight_study(work, 1:3, [1 2 4 8], false);
    case 'curved',    results = straight_study(work, 1:3, [1 2 4 8], true);
    case 'extension', results = extension_study(work, [1 2 4 8]);
    case 'p3diag',    results = p3_diagnostic(work);
    case 'localize',  results = localize_diagnostic(work);
    case 'b1sweep',   results = b1_sweep(work);
    case 'b1quad',    results = b1_quadrature(work);
    otherwise, error('unknown mode %s', mode);
end
end

% ---------------------------------------------------------------------------
function results = straight_study(work, porders, levels, curved)
A = 0.05;   % curved interface x = 0.5 + A sin(pi y)
label = 'straight'; if curved, label = 'curved'; end
results = struct('p', {}, 'N', {}, 'h1', {}, 'h2', {}, 'dofs', {}, 'e1', {}, 'e2', {}, 'e', {}, ...
    'gapmax', {}, 'gaprms', {}, 'gapbound', {}, 'newton', {});
build = fullfile(work, 'build');
for p = porders
    for N = levels
        [pde, mesh] = base_problem(p, N, fullfile(work, sprintf('p%d_N%d', p, N)), build, [2*N 4*N], [3*N 6*N], 1.0);
        gap = struct('max', 0, 'rms', 0, 'bound', 0);
        if curved
            [mesh, gap] = curve_interface(mesh, p, A, 1/(4*N), 1/(6*N));
        end
        out = phase9_fullsolve('run', pde, mesh);
        assert(out.status == 0, '%s p=%d N=%d solver failed', label, p, N);
        assert(has_both_directions(out.log), 'type-1 donor search did not run in both directions');
        [e1, nd1] = l2error(out.sol{1}, mesh{1}.dgnodes, out.master{1}.xpe, p);
        [e2, nd2] = l2error(out.sol{2}, mesh{2}.dgnodes, out.master{2}.xpe, p);
        r = struct('p', p, 'N', N, 'h1', 1/(4*N), 'h2', 1/(6*N), 'dofs', nd1 + nd2, 'e1', e1, 'e2', e2, ...
            'e', sqrt(e1^2 + e2^2), 'gapmax', gap.max, 'gaprms', gap.rms, 'gapbound', gap.bound, ...
            'newton', out.history.residuals(end));
        results(end+1) = r; %#ok<AGROW>
        fprintf('PHASE9 conv %s p=%d N=%d h1=%.5f h2=%.5f dofs=%d e1=%.6e e2=%.6e e=%.6e gapmax=%.3e gaprms=%.3e gapbound=%.3e newton=%.3e\n', ...
            label, p, N, r.h1, r.h2, r.dofs, e1, e2, r.e, gap.max, gap.rms, gap.bound, r.newton);
    end
end
% Rates between successive levels (h halves) and the predeclared criteria.
ok = true;
for p = porders
    rows = results([results.p] == p);
    for k = 2:numel(rows)
        rate = log2([rows(k-1).e1 rows(k-1).e2 rows(k-1).e] ./ [rows(k).e1 rows(k).e2 rows(k).e]);
        fprintf('PHASE9 conv %s p=%d N=%d->%d rates: model1 %.3f model2 %.3f union %.3f\n', ...
            label, p, rows(k-1).N, rows(k).N, rate);
        if k == numel(rows)
            target = p + 1 - 0.2; if curved, target = p + 1 - 0.3; end
            pass = all(rate >= target);
            ok = ok && pass;
            fprintf('PHASE9 conv %s p=%d last-level rates >= %.1f: %s\n', label, p, target, tf(pass));
        end
        if curved
            gapRate = log2(rows(k-1).gapmax / rows(k).gapmax);
            fprintf('PHASE9 conv %s p=%d N=%d->%d geometry gap rate %.3f\n', label, p, rows(k-1).N, rows(k).N, gapRate);
        end
    end
    if curved
        within = all([rows.gapmax] <= [rows.gapbound]);
        ok = ok && within;
        fprintf('PHASE9 conv curved p=%d measured gap <= analytic bound at every level: %s\n', p, tf(within));
    end
end
assert(ok, 'phase9_convergence %s: a predeclared criterion failed', label);
fprintf('PASS phase9_convergence %s\n', label);
end

% ---------------------------------------------------------------------------
function results = extension_study(work, levels)
% Model 1 on [0,0.5]x[0,1.125] (squaremesh(4N, 9N), h1 = 1/(8N)); model 2 as above.
% Owner points of Gamma^1 with y > 1 have no covering donor face and use polynomial
% extrapolation of the donor element (identified from pzeta in unit_nonmatching_setup).
% The uncovered strip has a fixed physical width while the donor element shrinks, so its
% reference-space extrapolation distance grows like 1/h and convergence there is not
% expected (Phase 9 prompt C5). Revision 1 contract, declared in phase9_report.md
% (Revision 1) after the Revision 0 data were known:
%   (a) Gamma^1 has exactly 3N points off its donor face (N interface faces above y = 1,
%       3 Gauss points each, p = 2) and Gamma^2 has none, in every solve;
%   (b) every solve succeeds and every solution value is finite (test-only check);
%   (c) the 1+1 and 2+2 solutions agree in global element order to 1e-10 relative;
%   (d) regression envelope (stability, not accuracy): max|u_h| <= 2 max|u| = 2 on both
%       models, and the total L2 error over both models never exceeds 1.5 times its value
%       on the coarsest level.
% The covered-region and strip errors are reported, not asserted.
p = 2;
results = struct('ranks', {}, 'N', {}, 'covered', {}, 'strip', {}, 'stripnorm', {}, 'total', {}, ...
    'umax', {}, 'off1', {}, 'off2', {}, 'sol', {});
build = fullfile(work, 'build');
ok = true;
for ranks = [1 2]
    for N = levels
        [pde, mesh] = base_problem(p, N, fullfile(work, sprintf('r%d_N%d', ranks, N)), ...
            build, [4*N 9*N], [3*N 6*N], 1.125);
        for m = 1:2, pde{m}.mpiprocs = ranks; end
        out = phase9_fullsolve('run', pde, mesh);
        assert(out.status == 0, 'extension N=%d ranks %d solver failed', N, ranks);
        assert(has_both_directions(out.log), 'type-1 donor search did not run in both directions');
        off1 = off_face_count(out.log, 1); off2 = off_face_count(out.log, 2);
        finite = all(isfinite(out.sol{1}(:))) && all(isfinite(out.sol{2}(:)));
        dg1 = mesh{1}.dgnodes;
        yc = squeeze(mean(dg1(:, 2, :), 1)); xc = squeeze(mean(dg1(:, 1, :), 1));
        covered1 = yc <= 1;
        strip = yc > 1 & xc >= 0.375;
        e1 = l2error(out.sol{1}(:, :, covered1), dg1(:, :, covered1), out.master{1}.xpe, p);
        e2 = l2error(out.sol{2}, mesh{2}.dgnodes, out.master{2}.xpe, p);
        es = l2error(out.sol{1}(:, :, strip), dg1(:, :, strip), out.master{1}.xpe, p);
        us = l2error(0*out.sol{1}(:, :, strip), dg1(:, :, strip), out.master{1}.xpe, p);
        et = sqrt(l2error(out.sol{1}, dg1, out.master{1}.xpe, p)^2 + e2^2);
        umax = max(max(abs(out.sol{1}(:, 1, :)), [], 'all'), max(abs(out.sol{2}(:, 1, :)), [], 'all'));
        r = struct('ranks', ranks, 'N', N, 'covered', sqrt(e1^2 + e2^2), 'strip', es, 'stripnorm', us, ...
            'total', et, 'umax', umax, 'off1', off1, 'off2', off2, 'sol', {out.sol});
        results(end+1) = r; %#ok<AGROW>
        countsOk = off1 == 3*N && off2 == 0;
        ok = ok && countsOk && finite && umax <= 2;
        fprintf(['PHASE9 conv extension %d+%d N=%d off-face Gamma1=%d (expected %d) Gamma2=%d finite=%d ' ...
                 'total=%.6e covered=%.6e strip=%.6e strip/||u||=%.3e max|u_h|=%.4f\n'], ...
            ranks, ranks, N, off1, 3*N, off2, finite, et, r.covered, es, es/us, umax);
    end
    rows = results([results.ranks] == ranks);
    bounded = all([rows.total] <= 1.5*rows(1).total);
    ok = ok && bounded;
    fprintf('PHASE9 conv extension %d+%d total error <= 1.5 x coarsest at every level: %s\n', ranks, ranks, tf(bounded));
end
for N = levels
    a = results([results.ranks] == 1 & [results.N] == N);
    b = results([results.ranks] == 2 & [results.N] == N);
    num = 0; den = 0;
    for m = 1:2
        num = max(num, max(abs(a.sol{m}(:) - b.sol{m}(:))));
        den = max(den, max(abs(a.sol{m}(:))));
    end
    same = num/den <= 1e-10;
    ok = ok && same;
    fprintf('PHASE9 conv extension N=%d 1+1 vs 2+2 global-order rel diff %.3e: %s\n', N, num/den, tf(same));
end
assert(ok, 'phase9_convergence extension: a contract item failed');
fprintf('PASS phase9_convergence extension\n');
end

function n = off_face_count(logtext, direction)
tok = regexp(logtext, sprintf('Gamma\\^%d \\(owners[^\\n]*?points off donor face = ([0-9]+)', direction), 'tokens', 'once');
assert(~isempty(tok), 'donor diagnostics for Gamma^%d not found', direction);
n = str2double(tok{1});
end

% ---------------------------------------------------------------------------
function results = b1_sweep(work)
% B1 diagnostic (no pass/fail): rates for p = 1..4 on three mesh pairs, N = 1, 2, 4, 8:
%   A  model 2 finer  (h1 = 1/(4N), h2 = 1/(6N))  - the failing C4 configuration
%   S  swapped        (h1 = 1/(6N), h2 = 1/(4N))  - model 1 finer
%   R  nested refinement (h1 = 1/(4N), h2 = 1/(8N)) - common refinement: every coarse
%      interface face is exactly two fine faces
results = struct('cfg', {}, 'p', {}, 'N', {}, 'e1', {}, 'e2', {});
cfgs = {'A', [2 4], [3 6]; 'S', [3 6], [2 4]; 'R', [2 4], [4 8]};
for c = 1:size(cfgs, 1)
    build = fullfile(work, 'build');
    for p = 1:4
        for N = [1 2 4 8]
            [pde, mesh] = base_problem(p, N, fullfile(work, sprintf('%s_p%d_N%d', cfgs{c, 1}, p, N)), build, ...
                cfgs{c, 2}*N, cfgs{c, 3}*N, 1.0);
            out = phase9_fullsolve('run', pde, mesh);
            assert(out.status == 0, 'b1 sweep %s p=%d N=%d failed', cfgs{c, 1}, p, N);
            e1 = l2error(out.sol{1}, mesh{1}.dgnodes, out.master{1}.xpe, p);
            e2 = l2error(out.sol{2}, mesh{2}.dgnodes, out.master{2}.xpe, p);
            results(end+1) = struct('cfg', cfgs{c, 1}, 'p', p, 'N', N, 'e1', e1, 'e2', e2); %#ok<AGROW>
            fprintf('PHASE9 b1 %s p=%d N=%d e1=%.6e e2=%.6e newton=%.2e\n', cfgs{c, 1}, p, N, e1, e2, ...
                out.history.residuals(end));
        end
        rows = results(strcmp({results.cfg}, cfgs{c, 1}) & [results.p] == p);
        fprintf('PHASE9 b1 %s p=%d rates model1', cfgs{c, 1}, p);
        fprintf(' %.3f', log2([rows(1:end-1).e1] ./ [rows(2:end).e1]));
        fprintf(' model2'); fprintf(' %.3f', log2([rows(1:end-1).e2] ./ [rows(2:end).e2])); fprintf('\n');
    end
end
end

% ---------------------------------------------------------------------------
function results = b1_quadrature(work)
% Face-quadrature sensitivity diagnostic. The mapped donor data is piecewise
% polynomial and may be discontinuous inside one owner face.
% donor data is piecewise polynomial on the donor faces, so it is discontinuous inside
% an owner face that spans several donor faces; a smooth integrand would make the
% error independent of pgauss once pgauss >= 2p. pgauss also sets the volume rule,
% whose integrands are smooth here.
results = struct('cfg', {}, 'p', {}, 'pg', {}, 'N', {}, 'e1', {}, 'e2', {});
cfgs = {'A', [2 4], [3 6], 3; 'R', [2 4], [4 8], 2; 'R', [2 4], [4 8], 3};
for c = 1:size(cfgs, 1)
    p = cfgs{c, 4};
    build = fullfile(work, 'build');
    for pg = unique([2*p 4*p 15])   % 15 is the largest pgauss in gaussnodes.bin
        for N = [1 2 4 8]
            [pde, mesh] = base_problem(p, N, fullfile(work, sprintf('%s_p%d_g%d_N%d', cfgs{c, 1}, p, pg, N)), ...
                build, cfgs{c, 2}*N, cfgs{c, 3}*N, 1.0, 1, pg);
            out = phase9_fullsolve('run', pde, mesh);
            assert(out.status == 0, 'b1quad %s p=%d pgauss=%d N=%d failed', cfgs{c, 1}, p, pg, N);
            e1 = l2error(out.sol{1}, mesh{1}.dgnodes, out.master{1}.xpe, p);
            e2 = l2error(out.sol{2}, mesh{2}.dgnodes, out.master{2}.xpe, p);
            results(end+1) = struct('cfg', cfgs{c, 1}, 'p', p, 'pg', pg, 'N', N, 'e1', e1, 'e2', e2); %#ok<AGROW>
            fprintf('PHASE9 b1quad %s p=%d pgauss=%d N=%d e1=%.6e e2=%.6e\n', cfgs{c, 1}, p, pg, N, e1, e2);
        end
        rows = results(strcmp({results.cfg}, cfgs{c, 1}) & [results.p] == p & [results.pg] == pg);
        fprintf('PHASE9 b1quad %s p=%d pgauss=%d rates model1', cfgs{c, 1}, p, pg);
        fprintf(' %.3f', log2([rows(1:end-1).e1] ./ [rows(2:end).e1]));
        fprintf(' model2'); fprintf(' %.3f', log2([rows(1:end-1).e2] ./ [rows(2:end).e2])); fprintf('\n');
    end
end
end

% ---------------------------------------------------------------------------
function results = localize_diagnostic(work)
% Diagnostic only: where do the p = 3 unequal-h error and the extension error live?
% (1) p = 3 unequal h: error in the elements touching the interface (one layer per
%     side) and in the rest of each model.
% (2) extension, p = 2: model-2 error, model-1 error below y = 0.75, model-1 error in
%     0.75 < y <= 1, strip error, and the mean of u_h and u over the strip.
results = [];
p = 3;
for N = [2 4 8]
    [pde, mesh] = base_problem(p, N, fullfile(work, sprintf('loc_p3_N%d', N)), fullfile(work, 'build_p3'), ...
        [2*N 4*N], [3*N 6*N], 1.0);
    out = phase9_fullsolve('run', pde, mesh);
    assert(out.status == 0);
    h = [1/(4*N) 1/(6*N)];
    for m = 1:2
        dg = mesh{m}.dgnodes;
        xc = squeeze(mean(dg(:,1,:), 1));
        band = abs(xc - 0.5) < h(m);
        eb = l2error(out.sol{m}(:,:,band), dg(:,:,band), out.master{m}.xpe, p);
        ei = l2error(out.sol{m}(:,:,~band), dg(:,:,~band), out.master{m}.xpe, p);
        fprintf('PHASE9 localize p3 N=%d model%d interface-layer=%.6e interior=%.6e\n', N, m, eb, ei);
    end
end
p = 2;
for N = [1 2 4 8]
    [pde, mesh] = base_problem(p, N, fullfile(work, sprintf('loc_ext_N%d', N)), fullfile(work, 'build_ext'), ...
        [4*N 9*N], [3*N 6*N], 1.125);
    out = phase9_fullsolve('run', pde, mesh);
    assert(out.status == 0);
    dg = mesh{1}.dgnodes;
    yc = squeeze(mean(dg(:,2,:), 1)); xc = squeeze(mean(dg(:,1,:), 1));
    lowSet = yc <= 0.75; midSet = yc > 0.75 & yc <= 1; strip = yc > 1 & xc >= 0.375;
    e2 = l2error(out.sol{2}, mesh{2}.dgnodes, out.master{2}.xpe, p);
    elow = l2error(out.sol{1}(:,:,lowSet), dg(:,:,lowSet), out.master{1}.xpe, p);
    emid = l2error(out.sol{1}(:,:,midSet), dg(:,:,midSet), out.master{1}.xpe, p);
    es = l2error(out.sol{1}(:,:,strip), dg(:,:,strip), out.master{1}.xpe, p);
    us = squeeze(out.sol{1}(:,1,strip)); ue = sin(pi*squeeze(dg(:,1,strip))).*sin(pi*squeeze(dg(:,2,strip)));
    fprintf('PHASE9 localize extension N=%d model2=%.6e model1(y<=0.75)=%.6e model1(0.75<y<=1)=%.6e strip=%.6e mean(u_h)=%.4e mean(u)=%.4e\n', ...
        N, e2, elow, emid, es, mean(us(:)), mean(ue(:)));
end
end

% ---------------------------------------------------------------------------
function results = p3_diagnostic(work)
% Diagnostic only (no pass/fail): p = 3 rates for (A) type 0 on matching meshes,
% (B) type 1 on the same matching meshes, (C) type 1 on the unequal-h meshes with a
% higher face quadrature (pgauss = 10 instead of 6), to locate the p = 3 rate loss.
p = 3; levels = [1 2 4 8];
cases = {'A type0 matching', 0, [2 4], []; 'B type1 matching', 1, [2 4], []; ...
         'C type1 unequal pgauss10', 1, [3 6], 10; 'D type1 unequal default', 1, [3 6], []};
results = struct('label', {}, 'N', {}, 'e1', {}, 'e2', {});
for c = 1:size(cases, 1)
    build = fullfile(work, sprintf('build%d', c));
    for N = levels
        n2 = cases{c, 3}*N;
        [pde, mesh] = base_problem(p, N, fullfile(work, sprintf('c%d_N%d', c, N)), build, [2*N 4*N], n2, 1.0, ...
            cases{c, 2}, cases{c, 4});
        out = phase9_fullsolve('run', pde, mesh);
        assert(out.status == 0, 'p3diag %s N=%d failed', cases{c, 1}, N);
        e1 = l2error(out.sol{1}, mesh{1}.dgnodes, out.master{1}.xpe, p);
        e2 = l2error(out.sol{2}, mesh{2}.dgnodes, out.master{2}.xpe, p);
        results(end+1) = struct('label', cases{c, 1}, 'N', N, 'e1', e1, 'e2', e2); %#ok<AGROW>
        fprintf('PHASE9 p3diag %s N=%d e1=%.6e e2=%.6e\n', cases{c, 1}, N, e1, e2);
    end
    rows = results(strcmp({results.label}, cases{c, 1}));
    fprintf('PHASE9 p3diag %s rates model1', cases{c, 1});
    fprintf(' %.3f', log2([rows(1:end-1).e1] ./ [rows(2:end).e1]));
    fprintf(' model2'); fprintf(' %.3f', log2([rows(1:end-1).e2] ./ [rows(2:end).e2])); fprintf('\n');
end
end

% ---------------------------------------------------------------------------
function [pde, mesh] = base_problem(p, N, casepath, build, n1, n2, ytop, interfacetype, pgauss)
if nargin < 8, interfacetype = 1; end
if nargin < 9, pgauss = []; end
pdeapp1;
pdeapp2;
pde{1}.modelnumber = 1; pde{2}.modelnumber = 2;
pde{1}.interfacefluxmap = 1; pde{2}.interfacefluxmap = 1;
[mesh{1}.p, mesh{1}.t] = squaremesh(n1(1), n1(2), 1, 1);
mesh{1}.p(1,:) = 0.5*mesh{1}.p(1,:);
mesh{1}.p(2,:) = ytop*mesh{1}.p(2,:);
mesh{1}.boundaryexpr = {@(q) abs(q(2,:))<1e-8, @(q) abs(q(1,:)-0.5)<1e-8, ...
    @(q) abs(q(2,:)-ytop)<1e-8, @(q) abs(q(1,:))<1e-8};
mesh{1}.boundarycondition = [1; 2; 1; 1];
mesh{1}.interfacecondition = [0; 1; 0; 0];
[mesh{2}.p, mesh{2}.t] = squaremesh(n2(1), n2(2), 1, 1);
mesh{2}.p(1,:) = 0.5*mesh{2}.p(1,:) + 0.5;
mesh{2}.boundaryexpr = {@(q) abs(q(2,:))<1e-8, @(q) abs(q(1,:)-1)<1e-8, ...
    @(q) abs(q(2,:)-1)<1e-8, @(q) abs(q(1,:)-0.5)<1e-8};
mesh{2}.boundarycondition = [1; 1; 1; 2];
mesh{2}.interfacecondition = [0; 0; 0; 1];
if ~exist(casepath, 'dir'), mkdir(casepath); end
for m = 1:2
    pde{m}.mpiprocs = 1;
    pde{m}.porder = p;
    pde{m}.interfacetype = interfacetype;
    if ~isempty(pgauss), pde{m}.pgauss = pgauss; end
    pde{m}.NLiter = 3;
    pde{m}.NLtol = 1e-10;
    pde{m}.linearsolvertol = 1e-12;
    pde{m}.matvectol = 1e-12;
    pde{m}.GMRESrestart = 400;
    pde{m}.linearsolveriter = 4000;
    pde{m}.datapath = string(casepath);
    pde{m}.builddir = string(build);
    pde{m}.buildpath = pde{m}.builddir;
    % Global-order DG nodes, given to preprocessing (it indexes them by elempart).
    f = facenumbering(mesh{m}.p, mesh{m}.t, 1, mesh{m}.boundaryexpr, []);
    mesh{m}.dgnodes = createdgnodes(mesh{m}.p, mesh{m}.t, f, [], [], p);
end
end

function [mesh, gap] = curve_interface(mesh, p, A, h1, h2)
% Map the DG nodes x' = x + A sin(pi y) w(x), w = x/0.5 (model 1), (1-x)/0.5 (model 2):
% the outer boundaries are fixed and both interfaces interpolate g(y) = 0.5 + A sin(pi y)
% at their own face nodes, so they differ between nodes by O(h^(p+1)).
dg = mesh{1}.dgnodes; dg(:,1,:) = dg(:,1,:) + A*sin(pi*dg(:,2,:)).*(dg(:,1,:)/0.5); mesh{1}.dgnodes = dg;
dg = mesh{2}.dgnodes; dg(:,1,:) = dg(:,1,:) + A*sin(pi*dg(:,2,:)).*((1-dg(:,1,:))/0.5); mesh{2}.dgnodes = dg;
% Independently sampled gap: y is unchanged by the map, so each side's interface is the
% degree-p Lagrange interpolant of g at its face nodes y0 + (y1-y0) t_i, with t_i the
% 1D master node positions.
t = unique(round(masternodes(p, 2, 1)*1e12)/1e12); t = t(:);
ys = linspace(0, 1, 20001).';
x1 = side_curve(ys, h1, t, A); x2 = side_curve(ys, h2, t, A);
d = x2 - x1;
gap.max = max(abs(d)); gap.rms = sqrt(mean(d.^2));
% Analytic bound (fixed before running): |g - I_p g| <= max|g^(p+1)| / (p+1)! *
% max|prod (y - y_i)| <= A pi^(p+1) / (p+1)! * p! h^(p+1) / 4 on each side.
gap.bound = A*pi^(p+1)*(h1^(p+1) + h2^(p+1))/(4*(p+1));
end

function x = side_curve(ys, h, t, A)
x = zeros(size(ys));
k = min(floor(ys/h), round(1/h) - 1);
for face = unique(k).'
    sel = k == face;
    yi = face*h + h*t;
    gi = 0.5 + A*sin(pi*yi);
    s = (ys(sel) - face*h)/h;
    L = ones(numel(s), numel(t));
    for i = 1:numel(t)
        for j = 1:numel(t)
            if j ~= i, L(:, i) = L(:, i).*(s - t(j))/(t(i) - t(j)); end
        end
    end
    x(sel) = L*gi;
end
end

% ---------------------------------------------------------------------------
function [err, ndof] = l2error(u, dgnodes, xpe, p)
% Physical-quadrature L2 error of the first component against sin(pi x) sin(pi y).
[npe, ~, ne] = size(dgnodes);
ndof = npe*ne;
assert(size(xpe, 1) == npe && npe == (p+1)^2, 'expected a Q%d quadrilateral master', p);
[a, b] = meshgrid(0:p, 0:p); a = a(:).'; b = b(:).';   % row vectors of exponents
V = (xpe(:,1).^a) .* (xpe(:,2).^b);
[g1, w1] = gauss01(p + 3);
[gx, gy] = meshgrid(g1, g1); gx = gx(:); gy = gy(:);
w = kron(w1, w1); w = w(:);
M = (gx.^a) .* (gy.^b);
Mx = (a .* gx.^max(a-1, 0)) .* (gy.^b);
My = (gx.^a) .* (b .* gy.^max(b-1, 0));
assert(isequal(size(V), [npe npe]) && rcond(V) > 1e-12, 'ill-formed Vandermonde');
phi = M / V; dphix = Mx / V; dphiy = My / V;
err = 0;
for e = 1:ne
    X = dgnodes(:, :, e);
    xg = phi*X;
    jac = (dphix*X(:,1)).*(dphiy*X(:,2)) - (dphiy*X(:,1)).*(dphix*X(:,2));
    uh = phi*u(:, 1, e);
    ue = sin(pi*xg(:,1)).*sin(pi*xg(:,2));
    err = err + sum(w.*abs(jac).*(uh - ue).^2);
end
err = sqrt(err);
end

function [x, w] = gauss01(n)
% Gauss-Legendre nodes and weights on [0,1] (Golub-Welsch).
k = 1:n-1;
beta = k./sqrt(4*k.^2 - 1);
[Q, D] = eig(diag(beta, 1) + diag(beta, -1));
[x, order] = sort(diag(D));
w = 2*Q(1, order).'.^2;
x = (x + 1)/2; w = w/2;
end

function yes = has_both_directions(logtext)
yes = contains(logtext, 'Non-matching donor search, Gamma^1') && contains(logtext, 'Non-matching donor search, Gamma^2');
end

function s = tf(b)
if b, s = 'PASS'; else, s = 'FAIL'; end
end
