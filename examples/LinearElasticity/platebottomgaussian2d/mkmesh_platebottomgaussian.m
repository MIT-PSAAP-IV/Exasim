function mesh = mkmesh_platebottomgaussian(nx, ny, growth)

if nargin < 1
    nx = 80;
end
if nargin < 2
    ny = 40;
end
if nargin < 3
    growth = 1.10;
end

if nx < 1 || ny < 1
    error("nx and ny must be positive integers.");
end
if growth <= 1.0
    error("growth must be greater than 1 to create bottom refinement.");
end

xmin = -0.5;
xmax = 0.5;
thickness = 0.2;
tol = 1e-8;

xv = linspace(xmin, xmax, nx + 1);
layers = growth .^ (0:ny);
yv = thickness * (layers - 1.0) / (layers(end) - 1.0);

[p, t] = quadgrid(xv, yv);

mesh.p = p';
mesh.t = t';
mesh.boundaryexpr = {...
    @(p) abs(p(2, :)) < tol, ...
    @(p) abs(p(1, :) - xmax) < tol, ...
    @(p) abs(p(2, :) - thickness) < tol, ...
    @(p) abs(p(1, :) - xmin) < tol};
mesh.periodicexpr = {};
mesh.curvedboundary = [0 0 0 0];
mesh.curvedboundaryexpr = {...
    @(p) 0 * p(1,:), @(p) 0 * p(1,:), ...
    @(p) 0 * p(1,:), @(p) 0 * p(1,:)};

end
