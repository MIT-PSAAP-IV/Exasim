function windturbine2d_plot_diagnostics(wt, outdir)
%WINDTURBINE2D_PLOT_DIAGNOSTICS Plot interface ordering and bad elements.

if nargin < 2 || isempty(outdir)
    outdir = wt.opts.workdir;
end
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

for i = 1:numel(wt.bladeLoops)
    figure(300+i); clf; hold on;
    q = wt.bladeLoops{i}.vertices;
    plot(q([1:end 1],1), q([1:end 1],2), '-o', 'MarkerSize', 3);
    step = max(1, floor(size(q,1)/30));
    for j = 1:step:size(q,1)
        text(q(j,1), q(j,2), sprintf('%d', j), 'FontSize', 8);
    end
    axis equal tight;
    title(sprintf('Blade %d interface point ordering', i));
    xlabel('x');
    ylabel('y');
    saveas(gcf, fullfile(outdir, sprintf('blade%d_interface_ordering.png', i)));
end

if isstruct(wt.background) && isfield(wt.background, 'p') && ~isempty(wt.background)
    [quality, worst] = local_element_quality(wt.background.p, wt.background.t, ...
        wt.background.elemtype);
    threshold = 0.2;
    bad = find(quality < threshold);

    figure(310); clf; hold on;
    local_patch(wt.background.p, wt.background.t, [0.88 1.0 0.88]);
    if ~isempty(bad)
        patch('faces', wt.background.t(bad,:), 'vertices', wt.background.p, ...
            'facecolor', [1.0 0.2 0.2], 'edgecolor', 'k', ...
            'FaceAlpha', 0.7, 'EdgeAlpha', 1);
    end
    axis equal tight;
    title(sprintf('Background elements with quality < %.2f', threshold));
    saveas(gcf, fullfile(outdir, 'background_low_quality_elements.png'));

    fprintf('Worst background element: %d, quality = %.6g\n', ...
        worst.element, worst.quality);
    fprintf('Elements below quality %.2f: %d of %d\n', ...
        threshold, numel(bad), numel(quality));
end
end

function local_patch(p, t, color)
patch('faces', t, 'vertices', p, 'facecolor', color, ...
    'edgecolor', [0.15 0.15 0.15], 'Linew', 0.5, ...
    'FaceAlpha', 1, 'EdgeAlpha', 1);
view(2);
end

function [q, worst] = local_element_quality(p, t, elemtype)
q = zeros(size(t, 1), 1);
if elemtype == 0
    for i = 1:size(t, 1)
        x = p(t(i,:), :);
        e12 = norm(x(2,:) - x(1,:));
        e23 = norm(x(3,:) - x(2,:));
        e31 = norm(x(1,:) - x(3,:));
        area = 0.5*abs(det([x(2,:) - x(1,:); x(3,:) - x(1,:)]));
        q(i) = 4*sqrt(3)*area/(e12^2 + e23^2 + e31^2);
    end
elseif elemtype == 1
    for i = 1:size(t, 1)
        x = p(t(i,:), :);
        q(i) = local_quad_scaled_jacobian(x);
    end
else
    error('Unsupported elemtype %d.', elemtype);
end
[worst.quality, worst.element] = min(q);
end

function q = local_quad_scaled_jacobian(x)
edges = x([2 3 4 1], :) - x;
lengths = sqrt(sum(edges.^2, 2));
sj = zeros(4, 1);
for j = 1:4
    a = x(j,:) - x(mod(j-2,4)+1,:);
    b = x(mod(j,4)+1,:) - x(j,:);
    sj(j) = det([a; b])/(norm(a)*norm(b));
end
q = min(abs(sj));
if any(lengths <= 0)
    q = 0;
end
end
