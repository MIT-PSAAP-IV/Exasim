function loops = windturbine2d_boundary_loops(mesh)
%WINDTURBINE2D_BOUNDARY_LOOPS Extract ordered vertex and DG boundary loops.
%
% The face extraction follows the boundaryplot convention: boundary faces
% are found from element faces with no neighboring element, then collected
% into continuous loops.  For every loop edge, the corresponding DG edge
% nodes are also returned in edge order.

p = mesh.p;
t = mesh.t;
if size(p, 2) ~= 2
    error('windturbine2d_boundary_loops expects mesh.p as np-by-2.');
end
if size(t, 2) ~= 4
    error('windturbine2d_boundary_loops currently supports quadrilateral meshes.');
end

face = [1 2; 2 3; 3 4; 4 1];
edges = [];
owners = [];
for e = 1:size(t, 1)
    for lf = 1:4
        edges = [edges; t(e, face(lf,:))]; %#ok<AGROW>
        owners = [owners; e lf]; %#ok<AGROW>
    end
end

sortedEdges = sort(edges, 2);
[~, ~, ic] = unique(sortedEdges, 'rows');
counts = accumarray(ic, 1);
isBoundary = counts(ic) == 1;
bedges = edges(isBoundary, :);
bowners = owners(isBoundary, :);

% Orient boundary edges so the adjacent element lies consistently on one
% side.  This mirrors boundaryplot's boundary-edge orientation logic.
for i = 1:size(bedges, 1)
    e = bowners(i, 1);
    lf = bowners(i, 2);
    other = setdiff(1:4, face(lf,:));
    opposite = t(e, other(1));
    v1 = p(bedges(i,2),:) - p(bedges(i,1),:);
    v2 = p(opposite,:) - p(bedges(i,1),:);
    if v1(1)*v2(2) - v1(2)*v2(1) > 0
        bedges(i,:) = bedges(i,[2 1]);
    end
end

edgeLoops = local_segcollect(bedges);
loops = repmat(struct('indices', [], 'vertices', [], 'dgnodes', [], ...
    'area', [], 'orientation', ""), numel(edgeLoops), 1);

for i = 1:numel(edgeLoops)
    ids = edgeLoops{i};
    if ids(1) == ids(end)
        ids = ids(1:end-1);
    end
    verts = p(ids, :);
    area = local_polygon_area(verts);
    loops(i).indices = ids(:);
    loops(i).vertices = verts;
    loops(i).dgnodes = local_loop_dgnodes(mesh, ids);
    loops(i).area = area;
    if area >= 0
        loops(i).orientation = "ccw";
    else
        loops(i).orientation = "cw";
    end
end

[~, order] = sort(abs([loops.area]), 'descend');
loops = loops(order);

end

function dg = local_loop_dgnodes(mesh, ids)
dg = [];
face = [1 2; 2 3; 3 4; 4 1];
for k = 1:numel(ids)
    a = ids(k);
    b = ids(mod(k, numel(ids)) + 1);
    found = false;
    for e = 1:size(mesh.t, 1)
        for lf = 1:4
            edge = mesh.t(e, face(lf,:));
            if all(edge == [a b]) || all(edge == [b a])
                q = mesh.dgnodes(mesh.perm(:,lf),:,e);
                if all(edge == [b a])
                    q = flipud(q);
                end
                if isempty(dg)
                    dg = q;
                else
                    dg = [dg; q(2:end,:)]; %#ok<AGROW>
                end
                found = true;
                break;
            end
        end
        if found
            break;
        end
    end
end
end

function loops = local_segcollect(e)
loops = {};
while ~isempty(e)
    current = e(1,1);
    loop = current;
    while true
        ix = find(e(:,1) == loop(end), 1);
        if isempty(ix)
            ix = find(e(:,2) == loop(end), 1);
            if isempty(ix)
                break;
            end
            next = e(ix,1);
        else
            next = e(ix,2);
        end
        loop = [loop next]; %#ok<AGROW>
        e(ix,:) = [];
        if next == current
            break;
        end
    end
    loops{end+1} = loop; %#ok<AGROW>
end
end

function area = local_polygon_area(p)
area = 0.5*sum(p(:,1).*p([2:end 1],2) - p(:,2).*p([2:end 1],1));
end
