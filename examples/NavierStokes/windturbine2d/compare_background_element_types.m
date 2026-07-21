%COMPARE_BACKGROUND_ELEMENT_TYPES Compare tri and quad background meshes.
%
% Add the Exasim Matlab frontend to the path before running this script.

cases = {"tri", "quad"};
plotdir = fullfile(tempdir, 'exasim_windturbine2d_element_type_compare');
if ~exist(plotdir, 'dir')
    mkdir(plotdir);
end

summary = struct([]);
for i = 1:numel(cases)
    etype = cases{i};
    opts = windturbine2d_options('plot', false, ...
        'gmsh', '/opt/homebrew/bin/gmsh', ...
        'workdir', fullfile(tempdir, ['exasim_windturbine2d_' char(etype)]), ...
        'backgroundElementType', etype, ...
        'backgroundMeshMode', 'graded', ...
        'backgroundMeshSizeFar', 1.0, ...
        'farfieldBoundarySpacing', 1.0, ...
        'backgroundMeshSizeNearBlade', 0.35, ...
        'backgroundMeshSizeTransition', 2.0, ...
        'backgroundMeshMinSize', 0.25, ...
        'backgroundMeshMaxSize', 1.2);

    wt = mkmesh_windturbine2d(opts);
    v = wt.info.validation;

    summary(i).backgroundElementType = etype; %#ok<SAGROW>
    summary(i).vertices = v.backgroundVertices;
    summary(i).elements = v.backgroundElements;
    summary(i).triangles = v.backgroundGmshTriangles;
    summary(i).quadrilaterals = v.backgroundGmshQuadrilaterals;
    summary(i).minQuality = v.backgroundMinElementQuality;
    summary(i).avgQuality = v.backgroundAvgElementQuality;
    summary(i).maxQuality = v.backgroundMaxElementQuality;
    summary(i).exasimMeshBuilt = isstruct(wt.background) && ...
        isfield(wt.background, 'dgnodes') && ~isempty(wt.background.dgnodes);
    summary(i).connection = v.mergeWarning;

    fprintf('%s background mesh\n', etype);
    fprintf('  vertices       = %d\n', summary(i).vertices);
    fprintf('  elements       = %d\n', summary(i).elements);
    fprintf('  Gmsh triangles = %d\n', summary(i).triangles);
    fprintf('  Gmsh quads     = %d\n', summary(i).quadrilaterals);
    fprintf('  quality min/avg/max = %.6g / %.6g / %.6g\n', ...
        summary(i).minQuality, summary(i).avgQuality, summary(i).maxQuality);
    fprintf('  Exasim DG mesh = %d\n', summary(i).exasimMeshBuilt);
    fprintf('  connection     = %s\n', summary(i).connection);

    figure(500+i); clf;
    local_plot_mesh(wt.background);
    axis equal;
    xlim([-opts.Rfar opts.Rfar]);
    ylim([-opts.Rfar opts.Rfar]);
    title([char(etype) ' background mesh']);
    saveas(gcf, fullfile(plotdir, [char(etype) '_background.png']));

    figure(510+i); clf;
    local_plot_final_mesh(wt);
    axis equal;
    xlim([-opts.Rfar opts.Rfar]);
    ylim([-opts.Rfar opts.Rfar]);
    title([char(etype) ' final connected mesh']);
    saveas(gcf, fullfile(plotdir, [char(etype) '_final.png']));
end

disp(struct2table(summary));
fprintf('element-type comparison plots written to %s\n', plotdir);

function local_plot_final_mesh(wt)
if isstruct(wt.mesh) && isfield(wt.mesh, 'p') && isfield(wt.mesh, 't')
    local_plot_mesh(wt.mesh);
    return;
end
if isstruct(wt.mesh) && isfield(wt.mesh, 'background') && isfield(wt.mesh, 'blades')
    hold on;
    local_plot_mesh(wt.mesh.background);
    for k = 1:numel(wt.mesh.blades)
        local_plot_mesh(wt.mesh.blades{k});
    end
    return;
end
local_plot_mesh(wt.background);
end

function local_plot_mesh(mesh)
if isempty(mesh) || ~isfield(mesh, 'p') || ~isfield(mesh, 't')
    return;
end
patch('faces', mesh.t, 'vertices', mesh.p, 'facecolor', [0.8,1,0.8], ...
    'edgecolor', 'k', 'Linew', 0.5, 'FaceAlpha', 1, 'EdgeAlpha', 1);
view(2);
end
