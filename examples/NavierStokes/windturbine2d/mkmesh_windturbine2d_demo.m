%MKMESH_WINDTURBINE2D_DEMO Generate a three-blade 2-D turbine mesh.
%
% Add the Exasim Matlab frontend to the path before running this script.

opts = windturbine2d_options();
opts.Nb = 3;
opts.Rrotor = 5;
opts.Rfar = 10;
opts.backgroundMeshMode = 'graded';
opts.backgroundElementType = "quad";
opts.backgroundMeshSizeFar = 1;
opts.farfieldBoundarySpacing = opts.backgroundMeshSizeFar;
opts.backgroundMeshSizeNearBlade = 0.05;
opts.backgroundMeshSizeTransition = 10.0;
opts.backgroundMeshSizeInterface = 0.05;
opts.backgroundMinElementSize = 0.05;
opts.backgroundMaxElementSize = 0.2;
opts.preserveInterfaceSegments = false;
opts.workdir = pwd();

opts.porder = 2;
opts.cmesh.nxw = 23;
opts.cmesh.nflr = 13;
opts.cmesh.nflf = 13;
opts.cmesh.nfuf = 15;
opts.cmesh.nfur = 15;
opts.cmesh.nr = 25;
opts.cmesh.TEC = 1;
opts.cmesh.spr = [10, 10, 10, 10, 10, 10, 10]*1;
opts.cmesh.yref = [0.1 0.3];
opts.cmesh.lw = 2;
opts.cmesh.ll = 0.2;
opts.cmesh.wakeopts = struct('expansionRatio', 1.5, ...
    'nstations', opts.cmesh.nxw + 1, 'verbose', false);

opts.plot = true;

wt = mkmesh_windturbine2d(opts);

disp(wt.info.validation);

pv = windturbine2d_poly2gmsh_points(wt.farLoop, wt.bladeLoops);
%figure(1); clf; plot(pv(:,1),pv(:,2),'o');

% poly2gmsh("windturbine.geo", pv, 0, 0.1, 0.5);
% pde1.gmsh = opts.gmsh;
% pde1.version = opts.gmshVersion;
% [p,t] = gmshcall(pde1, "windturbine", 2, 1);
% figure(1);clf;
% plot(pv(:,1),pv(:,2),'o');
% hold on;
% simpplot(p',t');

figure(2); clf; hold on;
plot(wt.farLoop([1:end 1],1), wt.farLoop([1:end 1],2), 'o-');
for i = 1:length(wt.bladeLoops)
    loop = wt.bladeLoops{i}.vertices;
    plot(loop([1:end 1],1), loop([1:end 1],2), 'o-');
end
axis equal tight;
title('Gmsh boundary loops');

% figure(3); clf;
% simpplot(wt.background.p, wt.background.t);
% axis equal tight;
% title('Gmsh background mesh');
