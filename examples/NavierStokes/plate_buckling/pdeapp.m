% Master driver for thermal buckling simulation
%
% Workflow:
%   1. Setup and mesh generation
%   2. Elasticity mesh deformation (optional)
%   3. NS solve + viscosity ramp (pdeapp_ns)
%   4. Helmholtz AV field + final NS solve
%   5. Final plots

run(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab', 'exasim_setup.m'));

%% ---- 1. Setup and mesh generation ----
porder = 2;
nx1 = 10; nxf = 30; ny = 20;
mesh = mkmesh_thermal_buckling(porder, nx1, nxf, ny);
figure(1); clf; meshplot(mesh);

%% ---- 2. Elasticity mesh deformation ----
bump_amp = 0.0;
bump_loc = 0.20;
bump_width = 0.05;

if abs(bump_amp) > 0
    [pde, ~] = initializeexasim();
    pde.porder = porder;
    mesh = pdeapp_el(mesh, pde, bump_amp, bump_loc, bump_width);
end

%% ---- 3. Navier-Stokes solve + viscosity ramp ----
pdeapp_ns;
mesh.dist = dist;

%% ---- 4. Helmholtz AV field + final NS solve ----
S0 = 0.2; gamma_hm = 1e3; lambda0 = 0.04; kappa0 = 4; eta = 0.9;
pdeapp_hm;

s = solhm(:,1,:);
s = s/max(s(:));
av = (s-S0).*(atan(gamma_hm*(s-S0))/pi + 0.5) - atan(gamma_hm)/pi + 0.5;
distav = tanh(mesh.dist*5);
av = lambda0*(av.*distav);
figure(); clf; scaplot(mesh, av);

mesh.vdg(:,1,:) = av;
mesh.udg = sol;
mesh.boundarycondition = [1, 2, 3, 5];
[pde,mesh,master,dmd] = preprocessing(pde,mesh);
runcode(pde, 1);
sol = fetchsolution(pde,master,dmd, fullfile(pde.datapath, 'dataout'));
figure(1); clf; scaplot(mesh, eulereval(sol, 'M',gam,Minf),[0 Minf],1); colorbar;

%% ---- 5. Final plots ----
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
