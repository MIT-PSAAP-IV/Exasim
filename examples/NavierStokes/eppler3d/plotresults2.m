
porder = 2;
n1 = 9;
n3 = 27;
nz = 32;
gamma = 1.4;
Ma = 0.09;
Minf = 0.09;
pinf = 1/(gam*Minf^2);          % freestream pressure

pde1.nd = 2;
pde1.porder = 2;
pde1.pgauss = 4;
pde1.elemtype = 1;
pde1.nodetype = 1;
master2d = Master(pde1);
mesh2d.xpe = mesh2d.plocal;
mesh2d.telem = mesh2d.tlocal;

% mesh = mkmesh_eppler3d(porder, 1, -2, nz, 0.1, mesh2d);
% mesh.xpe = master.xpe;

base = fullfile('tmp', 'epplerdns');

reavg = readbin(fullfile(base, 'sra', 'case3', 'sol2davg_step_0.bin'));
reavg = reshape(reavg, n1, 30, []);
faavg = FavreAverages(reavg);
figure(1); clf; scaplot(mesh2d, faavg(:,44,:), [],2);
xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('Kinetic turbulent energy');
colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

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


[n1, n2, n3, nsteps, xf] = read_rank(fullfile(base, 'outbou', 'case4', 'outbouxdg.bin'));
[n1, n2, n3, nsteps, nlf] = read_rank(fullfile(base, 'outbou', 'case4', 'outboundg.bin'));
[n1, n2, n3, nsteps, uhf] = read_rank(fullfile(base, 'outbou', 'case4', 'outbouuhmean.bin'));
[n1, n2, n3, nsteps, udgf] = read_rank(fullfile(base, 'outbou', 'case4', 'outbouudgmean.bin'));

% [n1, n2, n3, nsteps, udgavg] = read_rank(fullfile(base, 'udgavg', 'case4', 'spanwiseudgavg.bin'));
% figure(3); clf; scaplot(mesh2d, eulereval3d(udgavg,'p',gamma,Ma), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('RMS of Pressure fluctuations');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% figure(4); clf; scaplot(mesh2d, reavg(:,9,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('pMean');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);

% xf = permute(xf, [1 3 2]);
% nlf = permute(nlf, [1 3 2]);
% uhf = permute(uhf, [1 3 2]);
% udgf = permute(udgf, [1 3 2]);

xm = squeeze(mean(xf, 1));
zx = xm(:,[3 1]);
[zx, ind] = sortrows(zx); 
nx = length(ind)/nz;

xf = permute(xf, [1 3 2]);
xf = xf(:,:,ind);
nlf = permute(nlf, [1 3 2]);
nlf = nlf(:,:,ind);
uhf = permute(uhf, [1 3 2]);
uhf = uhf(:,:,ind);
udgf = permute(udgf, [1 3 2]);
udgf = udgf(:,:,ind);

p1=porder+1;
xa = squeeze(mean(reshape(xf, [p1 p1 3 nx nz]), [2,5]));
x = squeeze(xa(:,1,:));
y = squeeze(xa(:,2,:));

na = squeeze(mean(reshape(nlf, [p1 p1 3 nx nz]), [2,5]));
n1 = squeeze(na(:,1,:));
n2 = squeeze(na(:,2,:));
n3 = squeeze(na(:,3,:));

ncu = size(uhf,2);
uha = squeeze(mean(reshape(uhf, [p1 p1 ncu nx nz]), [2,5]));

nc = size(udgf,2);
udga = squeeze(mean(reshape(udgf, [p1 p1 nc nx nz]), [2,5]));

xyMidChord = [0.0 0.0; 0.8 0.015; 1.0 0.0];
lower = lowerSurfaceMask(x,y,1, xyMidChord);
upper = find(sum(lower,1)==0);
lower = find(sum(lower,1)>0);
xl = x(:,lower);
yl = y(:,lower);
xu = x(:,upper);
yu = y(:,upper);

figure(1); clf; 
plot(xl, yl, '-b');  hold on; plot(xu, yu, '-r'); 
axis tight; axis equal;

[p, txx, txy, txz, tyy, tyz, tzz, Qx, Qy, Qz] = nsfluxcart3d(uha(:,1:5,:), udga(:,6:end,:), 1.4, param(2), 0.72);
p = squeeze(p); pl = p(:,lower); pu = p(:,upper);
txx = squeeze(txx); 
txy = squeeze(txy); 
tyy = squeeze(tyy);

cp = -2*(squeeze(p)-pinf);
t1 = squeeze(txx).*n1 + squeeze(txy).*n2;
t2 = squeeze(txy).*n1 + squeeze(tyy).*n2;
% Skin shear stress 
cf = -2*(t1.*n2 - t2.*n1);
%cf = 2*(txy.*(n2.^2-n1.^2) + txx.*n1.*n2 - tyy.*n1.*n2);

cfl = cf(:,lower);
cfu = cf(:,upper);
figure(1); clf; 
plot(xu(2,:), cfu(2,:)); hold on;
plot(xl(2,:), cfl(2,:)); axis tight;
figure(2); clf; 
plot(xu(2,:), cp(2,upper)); hold on;
plot(xl(2,:), cp(2,lower)); axis tight;

[Cx,Cy] = surfaceForceCoefficients(cp,cf,x,y,master2d);
CD =  Cx*cos(alpha) + Cy*sin(alpha);
CL = -Cx*sin(alpha) + Cy*cos(alpha);

figure(1); clf; 
plot(xu(2,:), txx(2,upper).*n1(2,upper)); hold on;
plot(xl(2,:), txx(2,lower).*n1(2,lower)); axis tight;


figure(1); clf; 
plot(xu(2,:), txx(2,upper)); hold on;
plot(xl(2,:), txx(2,lower)); axis tight;

figure(1); clf; 
plot(xu(2,:), tyy(2,upper)); hold on;
plot(xl(2,:), tyy(2,lower)); axis tight;

figure(1); clf; 
plot(xu(2,:), txy(2,upper)); hold on;
plot(xl(2,:), txy(2,lower)); axis tight;

figure(1); clf; 
plot(xu(2,:), t1(2,upper)); hold on;
plot(xl(2,:), t1(2,lower)); axis tight;

figure(1); clf; 
plot(xu(2,:), t2(2,upper)); hold on;
plot(xl(2,:), t2(2,lower)); axis tight;

figure(1); clf; 
plot(xu(2,:), n1(2,upper)); hold on;
plot(xl(2,:), n1(2,lower)); axis tight;

figure(1); clf; 
plot(xu(2,:), n2(2,upper)); hold on;
plot(xl(2,:), n2(2,lower)); axis tight;

figure(2); clf; plot(x, cf); axis tight;
figure(3); clf; plot(x, squeeze(Qx), '.'); axis tight;

[xpe,telem,xpf,tface,perm] = masternodes(porder,3,1);

nref = 1;
xmin = -0.05; xmax = 1.0;
pres = eulereval3d(uhf(:,1:5,:),'p',gam,Minf)/pinf;
cmin = min(pres(:)); cmax = max(pres(:));

figure(1);clf;
plotudg(xf,pres,xpf,tface,porder,[],nref); colorbar;
axis equal; axis tight; view(3);

figure(1);clf;
plotudg(xf,pres,xpf,tface,porder,[],nref); colorbar; view(3);
hold on;
xt = xf; xt(:,3,:) = xt(:,3,:) + 0.03;
plotudg(xt,pres,xpf,tface,porder,[cmin cmax],nref, xmin, xmax);
xt(:,3,:) = xt(:,3,:) + 0.03;
plotudg(xt,pres,xpf,tface,porder,[cmin cmax],nref, xmin, xmax);
xt(:,3,:) = xt(:,3,:) + 0.03;
plotudg(xt,pres,xpf,tface,porder,[cmin cmax],nref, xmin, xmax);
xt(:,3,:) = xt(:,3,:) + 0.03;
plotudg(xt,pres,xpf,tface,porder,[cmin cmax],nref, xmin, xmax);
axis equal; axis tight; colormap('jet'); colorbar;
set(gca,'FontSize',16);
%exportgraphics(gca,"pressure16" + ".png",'Resolution',300);

% UDG = getsolution("dataout/outudg_t4000",dmd,n3);
% UDGavg = getmeansolution("dataout/outreavg",dmd,n3);
% [UDGspm,UDG2d] = spanwiseaverage(UDGavg,n1,nz,UDG,2,10);
%
% pde.visscalars = {"qcrit", 1, "u", 2, "p", 3};
% pde.visvectors = {};
% pde.paraview = "/Applications/ParaView-6.0.0.app/Contents/MacOS/paraview";
%
% qcrit = qcriterion(UDG);
% ind = qcrit(:)>2e2;
% qcrit(ind)=2e2;
% ind = qcrit(:)<-200;
% qcrit(ind)=-200;
%
% visfield = qcrit;
% visfield(:,2,:) = UDG(:,2,:)./UDG(:,1,:);
% visfield(:,3,:) = eulereval3d(UDG,'p',gamma,Ma);
% vis(visfield,pde,mesh);
%
% figure(2); clf; scaplot(mesh2d, UDGspm(:,2,:)./UDGspm(:,1,:),[],2);
% colormap('jet'); colorbar; axis([-0.05 1.35 -0.2 0.2]); set(gca,'FontSize',20);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average density');
% exportgraphics(gca, "density.png",'Resolution',300);
%
% figure(3); clf; scaplot(mesh2d, eulereval3d(UDGspm,'p',gamma,Ma),[],2);
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average pressure');
% exportgraphics(gca, "pressure.png",'Resolution',300);
%
% figure(4); clf; scaplot(mesh2d, eulereval3d(UDGspm,'vm',gamma,Ma),[],2);
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average velocity magnitude');
% exportgraphics(gca, "velocity.png",'Resolution',300);
%
% figure(5); clf; scaplot(mesh2d, eulereval3d(UDG2d,'u',gamma,Ma),[-0.1 1.5],2,1);
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('instantaneous horizontal velocity');
% exportgraphics(gca, "horizontalvelocity.png",'Resolution',300);
%
% figure(6); clf; scaplot(mesh2d, eulereval3d(UDG2d,'w',gamma,Ma),[-0.1 0.1],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('instantaneous spwanwise velocity');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% figure(7); clf; scaplot(mesh2d, eulereval3d(UDG2d,'vm',gamma,Ma),[0 1.5],2);
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('instantaneous velocity magnitude');
%
% figure(6); clf; scaplot(mesh2d, reavg(:,6,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average horizontal velocity');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
%
% figure(6); clf; scaplot(mesh2d, reavg(:,7,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average vertical velocity');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
%
% figure(6); clf; scaplot(mesh2d, reavg(:,8,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average spanwise velocity');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
%
% figure(6); clf; scaplot(mesh2d, reavg(:,9,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average pressure');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
%
% figure(6); clf; scaplot(mesh2d, reavg(:,17,:)-reavg(:,6,:).^2, [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average uu');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
%
% figure(6); clf; scaplot(mesh2d, reavg(:,18,:)-reavg(:,7,:).^2, [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average vv');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
%
% figure(6); clf; scaplot(mesh2d, reavg(:,19,:)-reavg(:,8,:).^2, [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average ww');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
%
% figure(6); clf; scaplot(mesh2d, reavg(:,20,:)-reavg(:,6,:).*reavg(:,7,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average uv');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);


% wid = 1;              % adjust to the wall boundary marker
% param = pde.physicsparam;
% param(2) = 4.6e5;
% xyMidChord = [0.0 0.0; 0.8 0.015; 1.0 0.0];
% [Cp,Cf,x,Cp2d,Cf2d,x2d,Ch,Ch2d,stress] = getsurfacedata(master2d,mesh2d,udgavg,param,wid,1,1,xyMidChord);
% figure(1);clf;plot(x(:,1),-Cf);axis tight;
% 
% figure(1);clf;plot(x(:,1),stress(:,1));axis tight;
