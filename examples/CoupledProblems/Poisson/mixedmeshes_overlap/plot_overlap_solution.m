function plot_overlap_solution(mesh, sol, triangleOnModel1)
% Show both overlapping topologies without allowing one patch set to hide the other.
if triangleOnModel1
    family = {'triangles', 'quadrilaterals'};
else
    family = {'quadrilaterals', 'triangles'};
end

for m = 1:2
    figure(m); clf;
    scaplot(mesh{m}, sol{m}(:,1,:), [0 1], 2, 1);
    axis on; axis equal; axis([0 1 0 1]);
    title(sprintf('Model %d (%s)', m, family{m}));
    colorbar;
end

figure(3); clf;
scaplot(mesh{1}, sol{1}(:,1,:), [0 1], 2, 0); hold on;
scaplot(mesh{2}, sol{2}(:,1,:), [0 1], 2, 0);
patch('Faces',mesh{1}.t','Vertices',mesh{1}.p','FaceColor','none', ...
    'EdgeColor',[0 0 0],'LineWidth',0.6);
patch('Faces',mesh{2}.t','Vertices',mesh{2}.p','FaceColor','none', ...
    'EdgeColor',[0.85 0.1 0.1],'LineWidth',0.6);
axis on; axis equal; axis([0 1 0 1]); colorbar;
title('Combined overlap');
h1 = plot(nan,nan,'-','Color',[0 0 0],'LineWidth',0.8);
h2 = plot(nan,nan,'-','Color',[0.85 0.1 0.1],'LineWidth',0.8);
legend([h1 h2], {'Model 1 mesh','Model 2 mesh'}, 'Location','southoutside');
end
