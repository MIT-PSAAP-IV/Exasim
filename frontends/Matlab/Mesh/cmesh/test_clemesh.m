function stats = test_clemesh(plotflag)
%TEST_CLEMESH Smoke test for clemesh using a six-block CLE airfoil mesh.
%
%   stats = test_clemesh()
%   stats = test_clemesh(true)
%
% The test constructs the six component meshes, maps them to a closed
% NACA0012 airfoil, connects them with clemesh, and reports basic topology
% and high-order geometry checks.

if nargin < 1
    plotflag = false;
end

porder = 2;
TEC = 15;
sps = [TEC, 1, 1, 1, 1, TEC, 1, 1, 1, 1, TEC];
spr = [10, 10, 10, 10, 10, 10, 10]*25;

[xf, yf] = local_naca0012(75);
[p, t] = clemeshparam6(37, 25, 25, 33, 41, 33, sps, spr, [0.01 0.05]);
[p, t, xdg] = clemeshmap(xf, yf, p, t, 10, 10, porder);
mesh = clemesh(p, t, xdg, porder);

area = local_signed_area(mesh.p, mesh.t, mesh.elemtype);
stats.verticesBefore = mesh.clemeshinfo.verticesBefore;
stats.verticesAfter = mesh.clemeshinfo.verticesAfter;
stats.mergedVertices = stats.verticesBefore - stats.verticesAfter;
stats.elements = size(mesh.t, 1);
stats.expectedElements = sum(cellfun(@(q) size(q, 1), t));
stats.minSignedArea = min(area);
stats.maxInterfaceMismatch = mesh.clemeshinfo.maxInterfaceMismatch;

assert(stats.elements == stats.expectedElements, ...
    'test_clemesh: element count changed during connection.');
assert(all(mesh.t(:) >= 1) && all(mesh.t(:) <= size(mesh.p, 1)), ...
    'test_clemesh: invalid connectivity.');
assert(stats.minSignedArea > 0, ...
    'test_clemesh: connected mesh contains inverted elements.');

fprintf('test_clemesh: vertices before = %d\n', stats.verticesBefore);
fprintf('test_clemesh: vertices after  = %d\n', stats.verticesAfter);
fprintf('test_clemesh: merged vertices = %d\n', stats.mergedVertices);
fprintf('test_clemesh: elements        = %d\n', stats.elements);
fprintf('test_clemesh: min area        = %.16e\n', stats.minSignedArea);
fprintf('test_clemesh: max mismatch    = %.16e\n', stats.maxInterfaceMismatch);

if plotflag
    figure(1); clf; hold on;
    for i = 1:6
        simpplot(p{i}, t{i});
    end
    axis equal tight;
    title('clemesh components');

    figure(2); clf;
    meshplot(mesh, 1);
    axis equal tight;
    title('connected clemesh');
end

end

function [xf, yf] = local_naca0012(n)
beta = linspace(0, pi, ceil(n/2))';
x = 0.5*(1 - cos(beta));
yt = 5*0.12*(0.2969*sqrt(x) - 0.1260*x - 0.3516*x.^2 + ...
    0.2843*x.^3 - 0.1036*x.^4);

xf = [flipud(x); x(2:end)];
yf = [flipud(yt); -yt(2:end)];
xf(end) = xf(1);
yf(end) = yf(1);
end

function area = local_signed_area(p, t, elemtype)
if elemtype == 0
    x1 = p(t(:,1),1); y1 = p(t(:,1),2);
    x2 = p(t(:,2),1); y2 = p(t(:,2),2);
    x3 = p(t(:,3),1); y3 = p(t(:,3),2);
    area = 0.5*((x2-x1).*(y3-y1) - (y2-y1).*(x3-x1));
else
    x = reshape(p(t',1), size(t,2), [])';
    y = reshape(p(t',2), size(t,2), [])';
    area = 0.5*sum(x.*y(:,[2:end 1]) - y.*x(:,[2:end 1]), 2);
end
end
