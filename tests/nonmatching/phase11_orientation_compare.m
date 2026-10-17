function report = phase11_orientation_compare(ranksPerModel, porder, perms)
%PHASE11_ORIENTATION_COMPARE M0 type-0/type-1 identity over interface orientations.
% Reorders the local vertices of every model-2 tetrahedron by each row of
% PERMS (default: the 12 even permutations of 1:4, which keep every element
% positively oriented), runs the matching tet/tet fixture with interface
% types 0 and 1, and requires a relative solution difference below 5e-12.
% The relative node permutation of every interface face (owner face node j
% of model 1 coincides with node pi(j) of the model-2 face) is recorded and
% printed. Odd permutations invert the elements (negative Jacobians), which
% Exasim does not support, so they are not used.

if nargin < 1, ranksPerModel = 1; end
if nargin < 2, porder = 1; end
if nargin < 3, perms = even_perms(4); end

tri = perms_all(3);   % orientation index = row of the sorted permutations of 1:3
seen = false(1, 6);
report = struct('perm', {}, 'relative', {}, 'orientations', {});
for k = 1:size(perms, 1)
    sigma = perms(k, :);
    conformal = phase11_preflight(0, ranksPerModel, 0, porder, sigma);
    nonmatching = phase11_preflight(0, ranksPerModel, 1, porder, sigma);
    relative = zeros(2, 1);
    for model = 1:2
        delta = nonmatching.sol{model} - conformal.sol{model};
        relative(model) = norm(delta(:))/max(norm(conformal.sol{model}(:)), 1);
    end
    orient = interface_orientations(conformal, tri);
    seen(orient) = true;
    fprintf('ORIENT perm [%s]: orientations {%s}, relative %.3e %.3e\n', ...
        num2str(sigma), num2str(unique(orient)), relative(1), relative(2));
    assert(max(relative) < 5e-12, ...
        '3D matching type-0/type-1 mismatch for perm [%s]: %.17e', num2str(sigma), max(relative));
    report(end+1) = struct('perm', sigma, 'relative', relative, 'orientations', unique(orient)); %#ok<AGROW>
end
fprintf('PASS phase11_orientation_compare tet %d+%d p%d: %d runs, orientations exercised {%s} of 6, max relative %.3e\n', ...
    ranksPerModel, ranksPerModel, porder, numel(report), num2str(find(seen)), max([report.relative], [], 'all'));
end

function orient = interface_orientations(result, tri)
% Relative vertex permutation of each interface face (x = 0.5), from the
% vertex mesh and the p=1 face-to-vertex map (valid for every porder).
[~, ~, ~, ~, fv] = masternodes(1, 3, 0);   % fv(:,f): local vertices of face f
faces = cell(2, 1);
for model = 1:2
    p = result.mesh{model}.p; t = result.mesh{model}.t;
    list = {};
    for e = 1:size(t, 2)
        for f = 1:size(fv, 2)
            x = p(:, t(fv(:, f), e))';
            if all(abs(x(:, 1) - 0.5) < 1e-12), list{end+1} = x; end %#ok<AGROW>
        end
    end
    faces{model} = list;
end
assert(numel(faces{1}) == numel(faces{2}) && ~isempty(faces{1}), 'interface face census failed');
orient = zeros(1, numel(faces{1}));
for i = 1:numel(faces{1})
    x1 = faces{1}{i};
    match = 0;
    for j = 1:numel(faces{2})
        if norm(mean(x1, 1) - mean(faces{2}{j}, 1)) < 1e-12, match = j; break; end
    end
    assert(match > 0, 'no matching model-2 face');
    x2 = faces{2}{match};
    q = zeros(1, 3);
    for a = 1:3
        b = find(all(abs(x2 - x1(a, :)) < 1e-12, 2));
        assert(numel(b) == 1, 'face vertices do not coincide');
        q(a) = b;
    end
    orient(i) = find(all(tri == q, 2));
end
end

function P = perms_all(n)
P = sortrows(perms(1:n));
end

function P = even_perms(n)
A = perms_all(n);
keep = false(size(A, 1), 1);
for i = 1:size(A, 1)
    q = A(i, :); inversions = 0;
    for a = 1:n, for b = a+1:n, inversions = inversions + (q(a) > q(b)); end, end
    keep(i) = mod(inversions, 2) == 0;
end
P = A(keep, :);
end
