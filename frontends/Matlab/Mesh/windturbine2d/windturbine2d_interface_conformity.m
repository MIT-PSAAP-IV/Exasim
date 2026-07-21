function report = windturbine2d_interface_conformity(p, t, bladeLoops, opts)
%WINDTURBINE2D_INTERFACE_CONFORMITY Verify background/blade edge matching.

tol = max(opts.mergeTolerance, opts.boundaryTolerance);
bedges = local_boundary_edges(t);
bp = p;

expected = 0;
matched = 0;
missing = 0;
extraOnInterface = 0;

for ib = 1:numel(bladeLoops)
    loop = bladeLoops{ib}.vertices;
    expected = expected + size(loop, 1);
    for i = 1:size(loop, 1)
        a = loop(i,:);
        b = loop(mod(i, size(loop,1)) + 1,:);
        if local_has_edge(bp, bedges, a, b, tol)
            matched = matched + 1;
        else
            missing = missing + 1;
        end
    end
end

for i = 1:size(bedges, 1)
    a = bp(bedges(i,1),:);
    b = bp(bedges(i,2),:);
    mid = 0.5*(a+b);
    if local_point_on_any_loop_segment(mid, bladeLoops, tol) && ...
            ~local_matches_any_loop_edge(a, b, bladeLoops, tol)
        extraOnInterface = extraOnInterface + 1;
    end
end

report.expectedInterfaceEdges = expected;
report.matchedInterfaceEdges = matched;
report.missingInterfaceEdges = missing;
report.extraInterfaceSubedges = extraOnInterface;
report.conforming = (missing == 0) && (extraOnInterface == 0);

fprintf(['Interface conformity: expected=%d, matched=%d, missing=%d, ' ...
    'extra subedges=%d, conforming=%d\n'], expected, matched, missing, ...
    extraOnInterface, report.conforming);
end

function bedges = local_boundary_edges(t)
if size(t, 2) == 3
    edges = [t(:,[1 2]); t(:,[2 3]); t(:,[3 1])];
elseif size(t, 2) == 4
    edges = [t(:,[1 2]); t(:,[2 3]); t(:,[3 4]); t(:,[4 1])];
else
    error('Unsupported element with %d vertices.', size(t, 2));
end
[~, ~, ic] = unique(sort(edges, 2), 'rows');
counts = accumarray(ic, 1);
bedges = edges(counts(ic) == 1, :);
end

function tf = local_has_edge(p, edges, a, b, tol)
tf = false;
for i = 1:size(edges, 1)
    e1 = p(edges(i,1),:);
    e2 = p(edges(i,2),:);
    if (norm(e1-a) <= tol && norm(e2-b) <= tol) || ...
            (norm(e1-b) <= tol && norm(e2-a) <= tol)
        tf = true;
        return;
    end
end
end

function tf = local_matches_any_loop_edge(a, b, bladeLoops, tol)
tf = false;
for ib = 1:numel(bladeLoops)
    loop = bladeLoops{ib}.vertices;
    for i = 1:size(loop, 1)
        c = loop(i,:);
        d = loop(mod(i, size(loop,1)) + 1,:);
        if (norm(a-c) <= tol && norm(b-d) <= tol) || ...
                (norm(a-d) <= tol && norm(b-c) <= tol)
            tf = true;
            return;
        end
    end
end
end

function tf = local_point_on_any_loop_segment(q, bladeLoops, tol)
tf = false;
for ib = 1:numel(bladeLoops)
    loop = bladeLoops{ib}.vertices;
    for i = 1:size(loop, 1)
        a = loop(i,:);
        b = loop(mod(i, size(loop,1)) + 1,:);
        if local_point_segment_distance(q, a, b) <= tol
            tf = true;
            return;
        end
    end
end
end

function d = local_point_segment_distance(q, a, b)
ab = b - a;
den = dot(ab, ab);
if den == 0
    d = norm(q-a);
    return;
end
s = max(0, min(1, dot(q-a, ab)/den));
d = norm(q - (a + s*ab));
end
