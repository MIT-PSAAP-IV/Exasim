function avField = computeavfield2dchem(xdgOrMesh, udg, ncu, maxdiv)
%COMPUTEAVFIELD2DCHEM Compute the chemistry AV field from a 2D solution.
%
%   avField = computeavfield2dchem(mesh)
%   avField = computeavfield2dchem(mesh, udg)
%   avField = computeavfield2dchem(xdg, udg)
%   avField = computeavfield2dchem(xdg, udg, ncu)
%
%   This mirrors the pointwise formulas in getavfield2dchem, but evaluates
%   them vectorized over every interpolation point and element of a computed
%   Exasim solution.
%
%   xdg has size npe x 2 x ne and stores physical coordinates. The radial
%   coordinate is xdg(:,2,:). udg has size npe x nc x ne and must contain
%   the conservative state followed by its gradients:
%
%       udg(:,1:ncu,:)             = u
%       udg(:,ncu+1:ncu+2*ncu,:)   = q
%
%   For the 2D ModelD layout nc = 3*ncu, so ncu is inferred from nc when it
%   is not supplied. The output avField has size npe x 1 x ne.

if nargin < 1
    error('computeavfield2dchem:NotEnoughInputs', ...
          'A mesh struct or xdg array is required.');
end

if isstruct(xdgOrMesh)
    mesh = xdgOrMesh;
    if ~isfield(mesh, 'dgnodes')
        error('computeavfield2dchem:MissingDgnodes', ...
              'The mesh struct must contain dgnodes.');
    end
    xdg = mesh.dgnodes;

    if nargin < 2 || isempty(udg)
        if ~isfield(mesh, 'udg')
            error('computeavfield2dchem:MissingUdg', ...
                  'The mesh struct must contain udg or udg must be supplied.');
        end
        udg = mesh.udg;
    end
else
    if nargin < 2
        error('computeavfield2dchem:MissingUdg', ...
              'The udg solution array is required when xdg is supplied.');
    end
    xdg = xdgOrMesh;
end

if ndims(xdg) ~= 3 || size(xdg, 2) < 2
    error('computeavfield2dchem:InvalidXdgSize', ...
          'xdg must have size npe x at-least-2 x ne.');
end
if ndims(udg) ~= 3
    error('computeavfield2dchem:InvalidUdgSize', ...
          'udg must have size npe x nc x ne.');
end
if size(xdg, 1) ~= size(udg, 1) || size(xdg, 3) ~= size(udg, 3)
    error('computeavfield2dchem:InconsistentSizes', ...
          'xdg and udg must have matching npe and ne dimensions.');
end

nc = size(udg, 2);
if nargin < 3 || isempty(ncu)
    if mod(nc, 3) ~= 0
        error('computeavfield2dchem:CannotInferNcu', ...
              'Cannot infer ncu because size(udg,2) is not divisible by 3.');
    end
    ncu = nc / 3;
end

if nargin < 4 || isempty(maxdiv)
    maxdiv = 100.0;
end
if ~isscalar(maxdiv) || ~isfinite(maxdiv) || maxdiv <= 0
    error('computeavfield2dchem:InvalidMaxDiv', ...
          'maxdiv must be a positive finite scalar.');
end

if ncu <= 3 || floor(ncu) ~= ncu
    error('computeavfield2dchem:InvalidNcu', ...
          'ncu must be a positive integer greater than 3.');
end
if nc < 3*ncu
    error('computeavfield2dchem:InsufficientComponents', ...
          'udg must contain at least ncu state components and 2*ncu gradient components.');
end

ns = ncu - 3;

alpha = 1.0e3;
rmin = 1.0e-3;
ymin = 1.0e-8;

rho_i = udg(:, 1:ns, :);
rho = sum(rho_i, 2);
rhou = udg(:, ns+1, :);
rhov = udg(:, ns+2, :);

q = udg(:, ncu+1:3*ncu, :);

drho_dz_i = -q(:, 1:ns, :);
drhou_dz = -q(:, ns+1, :);

drho_dr_i = -q(:, ncu+1:ncu+ns, :);
drhov_dr = -q(:, ncu+ns+2, :);

drho_dz = sum(drho_dz_i, 2);
drho_dr = sum(drho_dr_i, 2);

rho = rmin + lmax(rho-rmin, alpha);
rhoinv = 1.0 ./ rho;

uz = rhou .* rhoinv;
ur = rhov .* rhoinv;

duz_dz = (drhou_dz - drho_dz .* uz) .* rhoinv;
dur_dr = (drhov_dr - drho_dr .* ur) .* rhoinv;

y = xdg(:, 2, :);
yreg = ymin + lmax(y-ymin, alpha);

divu = duz_dz + dur_dr + ur ./ yreg;
comp = -divu;

avField = limiting(comp, 0, maxdiv, alpha, 0);

end
