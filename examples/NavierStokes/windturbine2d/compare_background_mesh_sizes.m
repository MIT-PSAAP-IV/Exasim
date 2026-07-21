%COMPARE_BACKGROUND_MESH_SIZES Compare coarse/medium/fine Gmsh backgrounds.
%
% Add the Exasim Matlab frontend to the path before running this script.

cases = {
    'coarse', 1.4, 1.4, 0.55, 2.5, 0.45, 1.6;
    'medium', 1.0, 1.0, 0.35, 2.0, 0.25, 1.2;
    'fine',   0.65, 0.65, 0.22, 1.5, 0.16, 0.8;
};

plotdir = fullfile(tempdir, 'exasim_windturbine2d_background_compare');
if ~exist(plotdir, 'dir')
    mkdir(plotdir);
end

summary = struct([]);
for i = 1:size(cases, 1)
    opts = windturbine2d_options('plot', false, ...
        'gmsh', '/opt/homebrew/bin/gmsh', ...
        'workdir', fullfile(tempdir, ['exasim_windturbine2d_' cases{i,1}]), ...
        'backgroundMeshMode', 'graded', ...
        'backgroundMeshSizeFar', cases{i,2}, ...
        'farfieldBoundarySpacing', cases{i,3}, ...
        'backgroundMeshSizeNearBlade', cases{i,4}, ...
        'backgroundMeshSizeTransition', cases{i,5}, ...
        'backgroundMeshMinSize', cases{i,6}, ...
        'backgroundMeshMaxSize', cases{i,7});

    wt = mkmesh_windturbine2d(opts);

    summary(i).name = cases{i,1}; %#ok<SAGROW>
    summary(i).backgroundMeshSizeFar = opts.backgroundMeshSizeFar;
    summary(i).farfieldBoundarySpacing = opts.farfieldBoundarySpacing;
    summary(i).backgroundMeshSizeNearBlade = opts.backgroundMeshSizeNearBlade;
    summary(i).backgroundMeshSizeTransition = opts.backgroundMeshSizeTransition;
    summary(i).backgroundMeshMinSize = opts.backgroundMeshMinSize;
    summary(i).backgroundMeshMaxSize = opts.backgroundMeshMaxSize;
    summary(i).farfieldPoints = wt.info.validation.farfieldPointCount;
    summary(i).vertices = wt.info.validation.backgroundVertices;
    summary(i).elements = wt.info.validation.backgroundElements;
    summary(i).minElementSize = wt.info.validation.backgroundMinElementSize;
    summary(i).maxElementSize = wt.info.validation.backgroundMaxElementSize;
    summary(i).mergeWarning = wt.info.validation.mergeWarning;

    fprintf('%s background mesh\n', summary(i).name);
    fprintf('  hFar        = %.6g\n', summary(i).backgroundMeshSizeFar);
    fprintf('  far spacing = %.6g\n', summary(i).farfieldBoundarySpacing);
    fprintf('  far points  = %d\n', summary(i).farfieldPoints);
    fprintf('  hNearBlade  = %.6g\n', summary(i).backgroundMeshSizeNearBlade);
    fprintf('  transition  = %.6g\n', summary(i).backgroundMeshSizeTransition);
    fprintf('  hMin/hMax   = %.6g / %.6g\n', summary(i).backgroundMeshMinSize, summary(i).backgroundMeshMaxSize);
    fprintf('  vertices    = %d\n', summary(i).vertices);
    fprintf('  elements    = %d\n', summary(i).elements);
    fprintf('  edge h min/max = %.6g / %.6g\n', summary(i).minElementSize, summary(i).maxElementSize);
    fprintf('  connection  = %s\n', summary(i).mergeWarning);

    figure(300+i); clf;
    local_plot_mesh(wt.background);
    axis equal;
    xlim([-opts.Rfar opts.Rfar]);
    ylim([-opts.Rfar opts.Rfar]);
    title([summary(i).name ' background mesh']);
    saveas(gcf, fullfile(plotdir, [summary(i).name '_background.png']));

    figure(400+i); clf;
    local_plot_farfield_boundary(wt.info.background.farfield);
    saveas(gcf, fullfile(plotdir, [summary(i).name '_farfield_boundary.png']));
end

disp(struct2table(summary));
fprintf('comparison plots written to %s\n', plotdir);

function local_plot_mesh(mesh)
patch('faces', mesh.t, 'vertices', mesh.p, 'facecolor', [0.8,1,0.8], ...
    'edgecolor', 'k', 'Linew', 0.5, 'FaceAlpha', 1, 'EdgeAlpha', 1);
view(2);
end

function local_plot_farfield_boundary(info)
theta = linspace(0, 2*pi, info.pointCount + 1)';
theta(end) = [];
p = info.radius * [cos(theta), sin(theta)];
plot(p([1:end 1],1), p([1:end 1],2), '-o', ...
    'LineWidth', 1, 'MarkerSize', 4);
axis equal;
title(sprintf('Far-field boundary: %d points, mean h = %.4g', ...
    info.pointCount, info.meanEdgeLength));
xlabel('x');
ylabel('y');
end
