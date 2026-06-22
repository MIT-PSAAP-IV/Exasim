function generate_inputfiles()
% Generate pdeapp.txt and binary input files from MATLAB pde+mesh structures.
% Usage:  generate_inputfiles  (from the plate_buckling directory)

cdir = pwd(); ii = strfind(cdir, "Exasim");
run(cdir(1:(ii+5)) + "/install/setpath.m");

[pde, ~] = initializeexasim();
pde.model = "ModelD";
pde.modelfile = "pdemodel_ns";
pde.platform = "cpu";
pde.mpiprocs = 16;
pde.hybrid = 1;
pde.porder = 2;
pde.pgauss = 2 * pde.porder;
pde.debugmode = 0;
pde.nd = 2;

gam = 1.451; Re = 9.84e5; Pr = 0.72; Minf = 7.7;
Tref = 477; Twall = 300; alpha_aoa = 0;
rinf = 1.0; ruinf = cos(alpha_aoa); rvinf = sin(alpha_aoa);
pinf = 1 / (gam * Minf^2); rEinf = 0.5 + pinf / (gam - 1);
Tinf = pinf / (gam - 1);

pde.physicsparam = [gam Re Pr Minf rinf ruinf rvinf rEinf Tinf Tref Twall];
pde.tau = 4.0;
pde.GMRESrestart = 250; pde.GMRESortho = 1;
pde.linearsolvertol = 1e-6; pde.linearsolveriter = 500;
pde.preconditioner = 1; pde.RBdim = 0; pde.ppdegree = 0;
pde.NLtol = 1e-6; pde.NLiter = 10; pde.matvectol = 1e-6;

% ---- mesh ----
nx1 = 10; nxf = 30; ny = 20;
mesh = mkmesh_thermal_buckling(pde.porder, nx1, nxf, ny);
master = Master(pde);

% ---- fit compilable boundary expressions ----
xmin = min(mesh.p(1,:)); xmax = max(mesh.p(1,:)); btol = 5e-4;

% Collect boundary face coordinates
% mesh.f(lf, ele) = bmark  (0=interior, 1..nbc=boundary marker)
nfe = size(mesh.f, 1);
ne  = size(mesh.f, 2);
bx = cell(4,1); by = cell(4,1);
for ele = 1:ne
    for lf = 1:nfe
        bmark = mesh.f(lf, ele);
        if bmark < 1 || bmark > 4, continue; end
        xdg = mesh.dgnodes(:, :, ele);
        pn  = master.perm(:, lf);
        pn(pn < 1 | pn > size(xdg,1)) = [];
        bx{bmark}(end+1:end+length(pn)) = xdg(pn, 1)';
        by{bmark}(end+1:end+length(pn)) = xdg(pn, 2)';
    end
end

[bx{3}, ix] = sort(bx{3}); by{3} = by{3}(ix);
[bx{4}, ix] = sort(bx{4}); by{4} = by{4}(ix);

% 8th-degree polynomial fits for curved boundaries
p3 = polyfit(bx{3}, by{3}, 8);
p4 = polyfit(bx{4}, by{4}, 8);

be1 = sprintf("abs(x-%16.12e)<%16.12e", xmax, btol);
if xmin < 0
    be2 = sprintf("abs(x+%16.12e)<%16.12e", -xmin, btol);
else
    be2 = sprintf("abs(x-%16.12e)<%16.12e", xmin, btol);
end
be3 = sprintf("abs(y-(%s))<%16.12e", poly2str(p3, "x"), btol);
be4 = sprintf("abs(y-(%s))<%16.12e", poly2str(p4, "x"), btol);

% Update boundaryexpr to use polyval (so facenumbering stays correct)
% but the pdeapp.txt will get the text2code-compatible strings
mesh.boundaryexpr = { ...
    @(p) abs(p(1,:)-xmax)<btol, ...
    @(p) abs(p(1,:)-xmin)<btol, ...
    @(p) abs(p(2,:)-polyval(p3,p(1,:)))<btol, ...
    @(p) abs(p(2,:)-polyval(p4,p(1,:)))<btol };

% ---- initial solution ----
dist = meshdist3(mesh.f, mesh.dgnodes, master.perm, [3]);
nm = 1e2;
mesh.vdg = zeros(size(mesh.dgnodes,1), 1, size(mesh.dgnodes,3));
mesh.vdg(:,1,:) = 0.005 * tanh(nm * dist);
mesh.dist = dist;
mesh.porder = pde.porder;
mesh.xpe = master.xpe;
mesh.telem = master.telem;

ui = [rinf ruinf rvinf rEinf];
UDG = initu(mesh, {ui(1), ui(2), ui(3), ui(4), 0,0,0,0,0,0,0,0});
UDG(:,2,:) = UDG(:,2,:) .* tanh(nm * dist);
UDG(:,3,:) = UDG(:,3,:) .* tanh(nm * dist);
TnearWall = Tinf * (Twall / Tref - 1) * exp(-nm * dist) + Tinf;
UDG(:,4,:) = TnearWall + 0.5 * (UDG(:,2,:).^2 + UDG(:,3,:).^2);
mesh.udg = UDG;

% ---- fill missing pde fields for writeinputfile ----
pde.ncu = 4; pde.ncw = 0;
[pde.nve, pde.neb] = size(mesh.t);
pde.nfb = 4 * pde.neb;
pde.torder = 1; pde.nstage = 1; pde.runmode = 0;
pde.modelnumber = 0; pde.time = 0; pde.dt = [0];
pde.matvecorder = 1; pde.precMatrixType = 0;
pde.externalparam = [];
pde.wmModelIDs = []; pde.wmBoundaries = []; pde.wmDistances = [];
pde.curvedboundaries = zeros(1, length(mesh.boundarycondition));
pde.curvedboundaryexprs = strings(1, length(mesh.boundarycondition));

% ---- generate pdeapp.txt + binary input files ----
outdir = fullfile(pwd(), "inputfiles");
if ~exist(outdir, 'dir'), mkdir(outdir); end
olddir = pwd(); cd(outdir);
writeinputfile("pdeapp.txt", pde, mesh);

% ---- patch boundary expressions with compilable forms ----
txt = fileread("pdeapp.txt");
oldExprs = convertHandlesToStrings(mesh.boundaryexpr);
newExprs = {be1, be2, be3, be4};
for i = 1:length(mesh.boundaryexpr)
    txt = strrep(txt, oldExprs(i), newExprs{i});
end
fid = fopen("pdeapp.txt", "w"); fwrite(fid, txt); fclose(fid);

cd(olddir);
fprintf("Generated: %s\n", outdir);
for i = 1:4
    fprintf("  Boundary %d: %s\n", i, newExprs{i});
end
end

% ---- helper: polynomial to expression string ----
function s = poly2str(p, var)
s = "";
n = length(p);
for i = 1:n
    deg = n - i;
    coeff = p(i);
    if coeff >= 0 && i > 1
        s = s + "+";
    end
    if abs(coeff) < 1e-30, continue; end
    if deg == 0
        s = s + sprintf("%.12e", coeff);
    elseif deg == 1
        s = s + sprintf("%.12e*%s", coeff, var);
    else
        s = s + sprintf("%.12e*pow(%s,%d)", coeff, var, deg);
    end
end
if s == "", s = "0.0"; end
end
