% Master driver for thermal buckling simulation
%
% Workflow:
%   1. Setup and mesh generation
%   2. Elasticity mesh deformation
%   3. NS solve + viscosity ramp (pdeapp_ns)
%   4. Final plots

%% ---- 1. Setup and mesh generation ----
cdir = pwd(); ii = strfind(cdir, "Exasim");
run(cdir(1:(ii+5)) + "/install/setpath.m");

[pde, ~] = initializeexasim();
pde.model = "ModelD";
pde.platform = "cpu";
pde.mpiprocs = 16;
pde.porder = 2;
pde.pgauss = 2 * pde.porder;
pde.hybrid = 1;
pde.debugmode = 0;
pde.nd = 2;

nx1 = 10; nxf = 30; ny = 20;
mesh = mkmesh_thermal_buckling(pde.porder, nx1, nxf, ny);
figure(1); clf; meshplot(mesh);

%% ---- 2. Elasticity mesh deformation ----
bump_amp = 0.24;
bump_loc = 0.20;
bump_width = 0.05;

if abs(bump_amp) > 0
    mesh = pdeapp_el(mesh, pde, bump_amp, bump_loc, bump_width);
end

%% ---- 3. Navier-Stokes solve + viscosity ramp ----
pdeapp_ns;
% writeinputfile("pdeapp_ns.txt", pde, mesh);


%% ---- 4. Helmholtz + NS solve ----
% HM parameters (may be overridden by master before calling pdeapp_hm)
S0 = 0.2; gamma = 1e3; lambda = 0.04; kappa = 4; eta = 0.9;
pdeapp_hm;

mesh.vdg = solhm;
mesh.udg = sol;
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1); % run C++ code
sol = fetchsolution(pde,master,dmd, fullfile(pde.datapath, 'dataout'));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;


%% ---- 5. Final plots ----
gam = 1.451;
Minf = 7.7;
Tref = 477;
pinf = 1 / (gam * Minf^2);
Tinf = pinf / (gam - 1);

mesh1 = mesh;
mesh1.dgnodes(:, 2, :) = -mesh.dgnodes(:, 2, :);

% Temperature
Tphys = Tref / Tinf * eulereval(sol, 't', gam, Minf);
figure(3); clf;
scaplot(mesh, Tphys, [], 1);
hold on;
scaplot(mesh1, Tphys, [], 1);
colorbar; colormap("jet"); axis on; axis equal; axis tight; set(gca, "FontSize", 16);
title("Temperature");
exportgraphics(gca, "temperature.png", "Resolution", 200);

% Pressure
R = 340.8;
rhophys = sol(:, 1, :) * 2.35e-3;
Pphys = rhophys .* Tphys * R;
figure(3); clf;
scaplot(mesh, Pphys, [], 1);
hold on;
scaplot(mesh1, Pphys, [], 1);
colorbar; colormap("jet"); axis on; axis equal; axis tight; set(gca, "FontSize", 16);
title("Pressure");
exportgraphics(gca, "pressure.png", "Resolution", 200);

% Mach
figure(3); clf;
scaplot(mesh, eulereval(sol, 'M', gam, Minf), [0 Minf], 1);
hold on;
scaplot(mesh1, eulereval(sol, 'M', gam, Minf), [0 Minf], 1);
colorbar; colormap("jet"); axis on; axis equal; axis tight; set(gca, "FontSize", 16);
title("Mach");
exportgraphics(gca, "mach.png", "Resolution", 200);
