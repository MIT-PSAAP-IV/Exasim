function [mesh, info] = windturbine2d_merge_components(background, blades, opts)
%WINDTURBINE2D_MERGE_COMPONENTS Merge vertex-conforming components.

if isempty(background)
    mesh = [];
    info.warning = "background mesh was not generated";
    return;
end

elemtypes = [background.elemtype; cellfun(@(m) m.elemtype, blades(:))];
if numel(unique(elemtypes)) > 1
    mesh = struct('background', background, 'blades', {blades});
    info.warning = "mixed element types cannot be represented by one scalar-elemtype Exasim mesh";
    info.elemtypes = elemtypes;
    return;
end

pall = background.p;
tall = background.t;
xdg = {background.dgnodes};
for i = 1:numel(blades)
    [pall, tall] = connectmesh(pall, tall, blades{i}.p, blades{i}.t, opts.mergeTolerance);
    xdg{end+1} = blades{i}.dgnodes; %#ok<AGROW>
end

mesh = mkmesh(pall, tall, opts.porder, background.bndexpr, background.elemtype, opts.nodetype);
mesh.dgnodes = cat(3, xdg{:});
info.warning = "";
info.vertices = size(mesh.p, 1);
info.elements = size(mesh.t, 1);
end
