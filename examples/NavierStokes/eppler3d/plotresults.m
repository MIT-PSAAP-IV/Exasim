
load dmd3d736.mat

n1 = 9;
n3 = 27;
nz = 18;
gamma = 1.4;
Ma = 0.09;

pde1.nd = 2;
pde1.porder = 2;
pde1.pgauss = 4;
pde1.elemtype = 1;
pde1.nodetype = 1;
master2d = Master(pde1);
mesh2d.xpe = mesh2d.plocal;
mesh2d.telem = mesh2d.tlocal;

nz = 18;
porder = 2;
mesh = mkmesh_eppler3d(porder, 1, -2, nz, 0.1, mesh2d);
mesh.xpe = master.xpe;

reavg = readbin("dataout/SpanwiseReynoldsAverages/eppler3dsweep7/paramcase_0007/sol2davg_step_0.bin");
reavg = reshape(reavg, n1, 30, []);
faavg = FavreAverages(reavg);

figure(1); clf; scaplot(mesh2d, reavg(:,6,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('uMean');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(2); clf; scaplot(mesh2d, reavg(:,7,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('vMean');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(3); clf; scaplot(mesh2d, reavg(:,8,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('wMean');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(4); clf; scaplot(mesh2d, reavg(:,9,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('pMean');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(1); clf; scaplot(mesh2d, -faavg(:,27,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('Kinetic turbulent energy');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(2); clf; scaplot(mesh2d, -faavg(:,28,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('Kinetic turbulent energy');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(3); clf; scaplot(mesh2d, -faavg(:,29,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('Kinetic turbulent energy');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(1); clf; scaplot(mesh2d, faavg(:,33,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('Kinetic turbulent energy');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(2); clf; scaplot(mesh2d, faavg(:,40,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('uRMS');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(3); clf; scaplot(mesh2d, faavg(:,44,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('RMS of Pressure fluctuations');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(6); clf; scaplot(mesh2d, reavg(:,17,:)-reavg(:,6,:).^2, [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average uu');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

UDG = getsolution("dataout/outudg_t4000",dmd,n3);
UDGavg = getmeansolution("dataout/outreavg",dmd,n3);
[UDGspm,UDG2d] = spanwiseaverage(UDGavg,n1,nz,UDG,2,10);

pde.visscalars = {"qcrit", 1, "u", 2, "p", 3};
pde.visvectors = {};
pde.paraview = "/Applications/ParaView-6.0.0.app/Contents/MacOS/paraview";

qcrit = qcriterion(UDG);
ind = qcrit(:)>2e2;
qcrit(ind)=2e2;
ind = qcrit(:)<-200;
qcrit(ind)=-200;

visfield = qcrit;
visfield(:,2,:) = UDG(:,2,:)./UDG(:,1,:);
visfield(:,3,:) = eulereval3d(UDG,'p',gamma,Ma);
vis(visfield,pde,mesh);

figure(2); clf; scaplot(mesh2d, UDGspm(:,2,:)./UDGspm(:,1,:),[],2);
colormap('jet'); colorbar; axis([-0.05 1.35 -0.2 0.2]); set(gca,'FontSize',20);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average density');
exportgraphics(gca, "density.png",'Resolution',300);

figure(3); clf; scaplot(mesh2d, eulereval3d(UDGspm,'p',gamma,Ma),[],2);
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average pressure');
exportgraphics(gca, "pressure.png",'Resolution',300);

figure(4); clf; scaplot(mesh2d, eulereval3d(UDGspm,'vm',gamma,Ma),[],2);
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average velocity magnitude');
exportgraphics(gca, "velocity.png",'Resolution',300);

figure(5); clf; scaplot(mesh2d, eulereval3d(UDG2d,'u',gamma,Ma),[-0.1 1.5],2,1);
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('instantaneous horizontal velocity');
exportgraphics(gca, "horizontalvelocity.png",'Resolution',300);

figure(6); clf; scaplot(mesh2d, eulereval3d(UDG2d,'w',gamma,Ma),[-0.1 0.1],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('instantaneous spwanwise velocity');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
figure(7); clf; scaplot(mesh2d, eulereval3d(UDG2d,'vm',gamma,Ma),[0 1.5],2);
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('instantaneous velocity magnitude');

figure(6); clf; scaplot(mesh2d, reavg(:,6,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average horizontal velocity');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(6); clf; scaplot(mesh2d, reavg(:,7,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average vertical velocity');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(6); clf; scaplot(mesh2d, reavg(:,8,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average spanwise velocity');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(6); clf; scaplot(mesh2d, reavg(:,9,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average pressure');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(6); clf; scaplot(mesh2d, reavg(:,17,:)-reavg(:,6,:).^2, [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average uu');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(6); clf; scaplot(mesh2d, reavg(:,18,:)-reavg(:,7,:).^2, [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average vv');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(6); clf; scaplot(mesh2d, reavg(:,19,:)-reavg(:,8,:).^2, [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average ww');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

figure(6); clf; scaplot(mesh2d, reavg(:,20,:)-reavg(:,6,:).*reavg(:,7,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average uv');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
