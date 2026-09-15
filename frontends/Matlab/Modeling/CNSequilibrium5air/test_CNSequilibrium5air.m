function report = test_CNSequilibrium5air(databaseFile)
%TEST_CNSEQUILIBRIUM5AIR Focused checks for the equilibrium-air CNS model.
%
% The tests are intentionally lightweight so they can run in Octave/MATLAB
% without invoking a full Exasim solve.  They verify the model conventions
% that are easiest to break: log-density material coordinates, xi-derivative
% conversion, temperature-gradient algebra, conductivity mixing, and 1-D/2-D/
% 3-D model hook sizes.

caseDir = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(databaseFile)
    databaseFile = fullfile(caseDir, '..', '..', '..', 'apps', ...
        'materialdatabases', 'equilibriumAir5logdensitymutationpp.dat');
end
databaseFile = char(databaseFile);
if ~exist(databaseFile, 'file')
    databaseFile = fullfile(fileparts(fileparts(fileparts(fileparts(caseDir)))), ...
        'apps', 'materialdatabases', 'equilibriumAir5logdensitymutationpp.dat');
end
if ~exist(databaseFile, 'file')
    error('Database file not found: %s', databaseFile);
end

addpath(caseDir, '-begin');
models = {'pdemodel', 'pdemodel_axial'};
db = local_read_material_dat(databaseFile);

rhoRef = 1.0;
uRef = 1000.0;
pRef = rhoRef*uRef^2;
eRef = uRef^2;
LRef = 1.0;
transportFactor = 1.0;
cChem = 0.0;
Twall = 300.0;
pOut = 1.0e5;
theta = 0.5;
mu = [rhoRef; uRef; pRef; eRef; LRef; transportFactor; cChem; Twall; pOut; theta];
report = struct();
report.model_sizes = zeros(0,8);

% Pick an interior table point away from boundaries.
ix = max(3, floor(numel(db.xi)/2));
ie = max(3, floor(numel(db.e)/2));
xi = db.xi(ix);
eDim = db.e(ie);
rhoDim = exp(xi);
rho = rhoDim/rhoRef;
e = eDim/eRef;

for im = 1:numel(models)
    pde = feval(models{im});
    for nd = 1:3
        vel = 0.1*(1:nd).';
        x = 0.25*ones(nd,1);
        if nd >= 2
            x(2) = 0.8;
        end
        u = [rho; rho*vel; rho*(e + 0.5*(vel.'*vel))];
        q = -0.01*reshape((1:((nd+2)*nd)).', [nd+2, nd]);
        w = local_interp(db, xi, eDim).';
        v = 0;
        eta = u;
        n = ones(nd,1); n = n/norm(n);
        tau = 2.0;
        uhat = u;

        state = pde.materialstate(u, q, w, v, x, 0, mu, eta);
        local_assert_close(state(1), xi, 1.0e-12, [models{im} ' materialstate xi']);
        local_assert_close(state(2), eDim, 1.0e-8, [models{im} ' materialstate e']);

        F = pde.flux(u, q, w, v, x, 0, mu, eta);
        S = pde.source(u, q, w, v, x, 0, mu, eta);
        FB = pde.fbou(u, q, w, v, x, 0, mu, eta, uhat, n, tau);
        FBH = pde.fbouhdg(u, q, w, v, x, 0, mu, eta, uhat, n, tau);
        UB = pde.ubou(u, q, w, v, x, 0, mu, eta, uhat, n, tau);
        if any(size(F) ~= [nd+2, nd]) || any(size(S) ~= [nd+2, 1])
            error('Unexpected residual size for %s nd=%d.', models{im}, nd);
        end
        if any(size(FB) ~= [nd+2, 8]) || any(size(FBH) ~= [nd+2, 8]) || any(size(UB) ~= [nd+2, 8])
            error('Unexpected boundary size for %s nd=%d.', models{im}, nd);
        end

        report.model_sizes(end+1,:) = [im, nd, size(F,1), size(F,2), size(S,1), size(FB,2), size(FBH,2), size(UB,2)]; %#ok<AGROW>
    end
end

% Derivative conversion p_rho = p_xi/rho_dim and T_rho = T_xi/rho_dim.
w0 = local_interp(db, xi, eDim);
dxi = min(db.xi(ix+1)-db.xi(ix), db.xi(ix)-db.xi(ix-1))*0.25;
wp = local_interp(db, log(rhoDim + rhoDim*(exp(dxi)-1)), eDim);
wm = local_interp(db, log(rhoDim - rhoDim*(1-exp(-dxi))), eDim);
drho = (rhoDim + rhoDim*(exp(dxi)-1)) - (rhoDim - rhoDim*(1-exp(-dxi)));
p_rho_fd = (wp(1) - wm(1))/drho;
T_rho_fd = (wp(2) - wm(2))/drho;
p_rho = w0(7)/rhoDim;
T_rho = w0(9)/rhoDim;
report.p_rho_relative_error = abs(p_rho_fd - p_rho)/max(1.0, abs(p_rho_fd));
report.T_rho_absolute_error = abs(T_rho_fd - T_rho);

% Temperature-gradient chain rule: grad(T)=T_xi*grad(rho)/rho+T_e*grad(e).
grho = 0.031*rho;
geDim = 2.0e4;
epsx = 1.0e-5;
Tp = local_interp(db, log(rhoDim + epsx*grho), eDim + epsx*geDim);
Tm = local_interp(db, log(rhoDim - epsx*grho), eDim - epsx*geDim);
gT_fd = (Tp(2) - Tm(2))/(2*epsx);
gT_chain = w0(9)*(grho/rhoDim) + w0(10)*geDim;
report.gradT_relative_error = abs(gT_fd - gT_chain)/max(1.0, abs(gT_fd));

% Conductivity mixing checks.
k0 = local_total_kappa(w0, 0.0);
k05 = local_total_kappa(w0, 0.5);
k1 = local_total_kappa(w0, 1.0);
local_assert_close(k0, w0(4), 0.0, 'kappa c=0');
local_assert_close(k05, w0(4) + 0.5*w0(5), 0.0, 'kappa c=0.5');
local_assert_close(k1, w0(4) + w0(5), 0.0, 'kappa c=1');
report.kappa = [k0, k05, k1];

fprintf('CNSequilibrium5air verification passed.\n');
fprintf('  database: %s\n', databaseFile);
fprintf('  p_rho relative error: %.3e\n', report.p_rho_relative_error);
fprintf('  T_rho absolute error: %.3e K m^3/kg\n', report.T_rho_absolute_error);
fprintf('  gradT relative error: %.3e\n', report.gradT_relative_error);
fprintf('  kappa(c=0,0.5,1) = [%.8g %.8g %.8g]\n', report.kappa);
end

function db = local_read_material_dat(filename)
fid = fopen(filename, 'r');
if fid < 0, error('Could not open %s.', filename); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
header = sscanf(fgetl(fid), '%f').';
if numel(header) ~= 5 || header(1) ~= 2
    error('Expected a two-state Exasim material database.');
end
nprop = header(2);
n1 = header(3);
n2 = header(4);
rows = fscanf(fid, '%f', [2+nprop, Inf]).';
if size(rows,1) ~= n1*n2
    error('Database row count does not match header.');
end
rows = sortrows(rows, [2 1]);
xi = unique(rows(:,1));
e = unique(rows(:,2));
props = zeros(numel(xi), numel(e), nprop);
for k = 1:size(rows,1)
    i = find(xi == rows(k,1), 1);
    j = find(e == rows(k,2), 1);
    props(i,j,:) = rows(k,3:end);
end
db = struct('xi', xi, 'e', e, 'props', props);
end

function w = local_interp(db, xi, e)
i = find(db.xi <= xi, 1, 'last');
j = find(db.e <= e, 1, 'last');
i = min(max(i,1), numel(db.xi)-1);
j = min(max(j,1), numel(db.e)-1);
tx = (xi - db.xi(i))/(db.xi(i+1)-db.xi(i));
te = (e - db.e(j))/(db.e(j+1)-db.e(j));
w00 = squeeze(db.props(i,j,:)).';
w10 = squeeze(db.props(i+1,j,:)).';
w01 = squeeze(db.props(i,j+1,:)).';
w11 = squeeze(db.props(i+1,j+1,:)).';
w = (1-tx)*(1-te)*w00 + tx*(1-te)*w10 + (1-tx)*te*w01 + tx*te*w11;
end

function k = local_total_kappa(w, c)
k = w(4) + c*w(5);
end

function local_assert_close(a, b, tol, label)
err = abs(a-b);
if err > tol
    error('%s mismatch: %.17g vs %.17g (err %.3e, tol %.3e).', label, a, b, err, tol);
end
end
