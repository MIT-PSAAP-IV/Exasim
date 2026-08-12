nz = 32;
gam = 1.4;                      % gas constant
Minf = 0.09;                    % freestream Mach number
tau = 10;                        % stabilization parameter
alpha = 6*pi/180;               % angle of attack
beta = 0;                       % sideslip angle
rinf = 1.0;                     % freestream density
ruinf = cos(alpha)*cos(beta);   % freestream x momentum
rvinf = sin(alpha);             % freestream y momentum
rwinf = cos(alpha)*sin(beta);   % freestream z momentum
pinf = 1/(gam*Minf^2);          % freestream pressure
rEinf = 0.5 + pinf/(gam-1);     % freestream energy
Re = 200000;                     % Reynolds number
Pr = 0.72;                      % Prandtl number

physicsparam = [gam Re Pr Minf rinf ruinf rvinf rwinf rEinf];
physicparams=repmat(physicsparam, [4 1]);
physicparams(:,2)=[1 2 3 4.6]'*1e5;

foilfile = fullfile(fileparts(mfilename('fullpath')), 'epp387_smoothed');
[xf,yf] = read_foil(foilfile);
TEC = 15;
sps = [TEC, 1, 1, 1, 1, TEC, 1, 1, 1, 1, TEC];
spr = [10, 10, 10, 10, 10, 10, 10]*70;
yref = [0.0025 0.008 0.02 0.036];
lw = 10;
ll = 10;
nxw = 21;
nflr = 11;
nflf = 11;
nfuf = 15;
nfur = 21;
nr   = 41;
mesh2d = clemesh_airfoil(xf, yf, nxw, nflr, nflf, nfuf, nfur, nr, sps, spr, yref, lw, ll, porder);
pde1.nd = 2;
pde1.porder = 2;
pde1.pgauss = 4;
pde1.elemtype = 1;
pde1.nodetype = 1;
master2d = Master(pde1);
mesh2d.xpe = mesh2d.plocal;
mesh2d.telem = mesh2d.tlocal;

base = "/Users/cuongnguyen/Documents/Exasim/tmp/epplerdns";

reynolds_averages(base, mesh2d, physicparams);
[xSurf, ySurf, Cp, Cf, CL, CD] = computeCpCf(base, physicparams);

wid = 1;
delta = 0.12;
m = 501;
alpha = 0;
computeBLthicknesses(base, mesh2d, master2d, physicparams, wid, delta, m, alpha);

base = "/Users/cuongnguyen/Documents/Exasim/tmp/epplerdns/outudg";
qcritvort(base, pde, mesh, dmd, mesh2d, master2d, physicparams);

% fileout = "/Users/cuongnguyen/Documents/Exasim/tmp/epplerdns/outudg/case2/qcrit.vtu";
% filein = "/Users/cuongnguyen/Documents/Exasim/tmp/epplerdns/outudg/case2/outudg_t4000";
% pde.paraview = "/Applications/ParaView-6.0.0.app/Contents/MacOS/paraview";
% UDG = paraviewqcrit(fileout, filein, pde, mesh, dmd);
% UDG = reshape(UDG, [master2d.npe master2d.npf 20 mesh2d.ne nz]);
% UDG2d = squeeze(UDG(:,2,:,:,16));

%[vgn, vgt, xgn, nlg] = velocityprofiles(base+"/sra/case1", mesh2d, master2d, wid, delta, m, alpha);

reavg = readReynoldsAverageFile(fullfile(base, 'sra', 'case1', 'sol2davg_step_0.bin'), master2d.npe, 30, mesh2d.ne);
[n1, n2, n3, nsteps, udgavg] = read_rank(fullfile(base, 'udgavg', 'case1', 'spanwiseudgavg.bin'));

[ugn, xgn, nlg] = surfacefieldalongnormal(mesh2d, master2d, udgavg, wid, delta, m, alpha);
[~, ~, vortz] = nsevalcart3d(ugn(:,1:5,:,:), ugn(:,6:20,:,:));
x1 = xgn(:,1,:,:);
x2 = xgn(:,2,:,:);
s = sqrt((x1 - x1(:,:,:,1)).^2 + (x2 - x2(:,:,:,1)).^2);
for j = 1:size(s,3)
  y = s(1,1,j,:); 
  vort =  vortz(1,1,j,:);
  [blt(j), dpt(j), mot(j), H(j), ve(j)] = BLthicknesses(y(:), vort(:));
end

% y = s(1,1,j,:); 
% vort =  vortz(1,1,j,:);
% [blt(j), dpt(j), mot(j), H(j), ve(j), vv] = BLthicknesses(y(:), vort(:));
% figure(2);clf;plot(y(:),vv(:),'o');
% 

xyMidChord = [0.0 0.0; 0.8 0.015; 1.0 0.0];
xs = xgn(1,1,:,1); xs=xs(:);
ys = xgn(1,2,:,1); ys=ys(:);
lower = lowerSurfaceMask(xs,ys,1,xyMidChord);
upper = ~lower;
xl = xs(lower);
xu = xs(upper);
bl = blt(lower);
bu = blt(upper);
dl = dpt(lower);
du = dpt(upper);
ml = mot(lower);
mu = mot(upper);
Hl = H(lower);
Hu = H(upper);
figure(1);clf;plot(xu,du);
figure(1);clf;plot(xu,mu);
figure(1);clf;plot(xu,Hu);

figure(1);clf;plot(xu,smoothing(du, 10));
figure(1);clf;plot(xu,smoothing(mu, 10));
figure(1);clf;plot(xu,smoothing(Hu, 10));

i1 = xl>0.022;
figure(1);clf;plot(xl(i1),smoothing(dl(i1), 10));
figure(2);clf;plot(xl(i1),smoothing(ml(i1), 10));
figure(3);clf;plot(xl(i1),smoothing(Hl(i1), 10));

utp2 = (ugn(:,23,:,:) - ugn(:,21,:,:).^2) + (ugn(:,24,:,:) - ugn(:,22,:,:).^2);

udgavg(:,21:25,:) = reavg(:,[6 7 17 18 20],:);

% utp2 = nlg(:,2,:).^2.*(ugn(:,23,:,:) - ugn(:,21,:,:).^2) + ...
%        nlg(:,1,:).^2.*(ugn(:,24,:,:) - ugn(:,22,:,:).^2) - ...
%        2*nlg(:,1,:).*nlg(:,2,:).*(ugn(:,25,:,:) - ugn(:,21,:,:).*ugn(:,22,:,:));

[vortx, vorty, vortz] = nsevalcart3d(ugn(:,1:5,:,:), ugn(:,6:20,:,:));
v = ugn(:,2,:,:)./ugn(:,1,:,:);
v(:,2,:,:) = ugn(:,3,:,:)./ugn(:,1,:,:);
[vgn, vgt] = normaltangentvelocity(v, nlg);

x1 = xgn(:,1,:,:);
x2 = xgn(:,2,:,:);
s = sqrt((x1 - x1(:,:,:,1)).^2 + (x2 - x2(:,:,:,1)).^2);
%figure(1);clf;plot(x1(:),x2(:),'o');

j = 720;
[x1(2,1,j,1) x2(2,1,j,1)]
a = vgt(2,1,j,:); a=a(:);
b = s(2,1,j,:);b=b(:);
vort =  vortz(2,1,j,:); vort = vort(:);
ut2 =  utp2(2,1,j,:); ut2 = ut2(:);
figure(1);clf;plot(b(:),a(:),'o');
figure(2);clf;plot(b(:),vort(:),'o');

h = b(2)-b(1);
v = [0; 0.5*(vort(1:end-1)+vort(2:end))*h];
v = cumsum(v);
dvort = (vort(2:end)-vort(1:end-1))/h;
dvort = abs([dvort; dvort(end)]);
figure(2);clf;plot(b(:),dvort(:).*b.^2,'o');

figure(1);clf;plot(b(:),a(:),'o',b(:),v(:),'s');

for i = 1:length(b)
  if (abs(vort(i))*b(i)<0.01*abs(v(i))) && (abs(dvort(i))*b(i)*b(i)<abs(v(i))) && (abs(v(i)) > 0.5)
    break;
  end
end
figure(1);clf;plot(b(i),v(i),'o',b(:),v(:),'-');
ve = v(i);
d = 1 - v(1:i)/ve;
d = sum(0.5*(d(1:end-1)+d(2:end))*h);
m = (1 - v(1:i)/ve).*v(1:i)/ve;
m = sum(0.5*(m(1:end-1)+m(2:end))*h);

[blt, dpt, mot, ve] = BLthicknesses(b, vort);

figure(1);clf;plot(ut2,b,'-');

vt2 = [0; 0.5*(ut2(1:end-1)+ut2(2:end))*h];
vt2 = cumsum(vt2);


P = polyfit(b(:),a(:),30);
dP = polyder(P);
ddP = polyder(dP);
f = polyval(P,b(:));
df = polyval(dP, b(:));
dfb = df(:).*b(:);
ddf = polyval(ddP, b(:));
ddfb2 = ddf(:).*b(:).^2;

for i = 1:length(dfb)
  if (abs(dfb(i))<0.03) && (abs(ddfb2(i)) < 0.1) && (abs(f(i)) > 0.9)
    break;
  end
end
i1 = abs(dfb)<0.05;
i2 = abs(ddfb2)<0.5;

figure(1);clf;plot(b(:),a(:),'o',b(:),f(:),'-','LineWidth',1);
figure(2);clf;plot(b(:),df(:).*b(:),'-','LineWidth',1);
figure(3);clf;plot(b(i1),dfb(i1),b(i2),ddfb2(i2),'-','LineWidth',1);
figure(4);clf;plot(b(i),a(i),'o',b(:),f(:),'-','LineWidth',1);

[x1(2,1,j,1) x2(2,1,j,1)]
figure(1);clf;plot(a(:),b(:),'o');
figure(2);clf;plot(c(:),b(1:end-1),'o');
figure(3);clf;plot(d(ind),e(ind),'o');


a = vgt(2,1,400,:);
b = s(2,1,400,:);
figure(1);clf;plot(a(:),b(:),'-');


% porder = 2;
% n1 = 9;
% n3 = 27;
% nz = 32;
% gamma = 1.4;
% Ma = 0.09;
% Minf = 0.09;
% pinf = 1/(gam*Minf^2);          % freestream pressure
% 
% pde1.nd = 2;
% pde1.porder = 2;
% pde1.pgauss = 4;
% pde1.elemtype = 1;
% pde1.nodetype = 1;
% master2d = Master(pde1);
% mesh2d.xpe = mesh2d.plocal;
% mesh2d.telem = mesh2d.tlocal;
% 
% % mesh = mkmesh_eppler3d(porder, 1, -2, nz, 0.1, mesh2d);
% % mesh.xpe = master.xpe;
% 
% base = fullfile('tmp', 'epplerdns');
% 
% reavg = readbin(fullfile(base, 'sra', 'case3', 'sol2davg_step_0.bin'));
% reavg = reshape(reavg, n1, 30, []);
% faavg = FavreAverages(reavg);
% figure(1); clf; scaplot(mesh2d, faavg(:,44,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('Kinetic turbulent energy');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% figure(1); clf; scaplot(mesh2d, reavg(:,6,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('uMean');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% figure(2); clf; scaplot(mesh2d, reavg(:,7,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('vMean');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% figure(3); clf; scaplot(mesh2d, reavg(:,8,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('wMean');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% figure(4); clf; scaplot(mesh2d, reavg(:,9,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('pMean');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% figure(1); clf; scaplot(mesh2d, -faavg(:,27,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('Kinetic turbulent energy');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% figure(2); clf; scaplot(mesh2d, -faavg(:,28,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('Kinetic turbulent energy');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% figure(3); clf; scaplot(mesh2d, -faavg(:,29,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('Kinetic turbulent energy');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% figure(1); clf; scaplot(mesh2d, faavg(:,33,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('Kinetic turbulent energy');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% figure(2); clf; scaplot(mesh2d, faavg(:,40,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('uRMS');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% figure(3); clf; scaplot(mesh2d, faavg(:,44,:), [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('RMS of Pressure fluctuations');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% figure(6); clf; scaplot(mesh2d, reavg(:,17,:)-reavg(:,6,:).^2, [],2);
% xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average uu');
% colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% 
% [n1, n2, n3, nsteps, xf] = read_rank(fullfile(base, 'outbou', 'case4', 'outbouxdg.bin'));
% [n1, n2, n3, nsteps, nlf] = read_rank(fullfile(base, 'outbou', 'case4', 'outboundg.bin'));
% [n1, n2, n3, nsteps, uhf] = read_rank(fullfile(base, 'outbou', 'case4', 'outbouuhmean.bin'));
% [n1, n2, n3, nsteps, udgf] = read_rank(fullfile(base, 'outbou', 'case4', 'outbouudgmean.bin'));
% 
% % [n1, n2, n3, nsteps, udgavg] = read_rank(fullfile(base, 'udgavg', 'case4', 'spanwiseudgavg.bin'));
% % figure(3); clf; scaplot(mesh2d, eulereval3d(udgavg,'p',gamma,Ma), [],2);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('RMS of Pressure fluctuations');
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% % 
% % figure(4); clf; scaplot(mesh2d, reavg(:,9,:), [],2);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('pMean');
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% % xf = permute(xf, [1 3 2]);
% % nlf = permute(nlf, [1 3 2]);
% % uhf = permute(uhf, [1 3 2]);
% % udgf = permute(udgf, [1 3 2]);
% 
% xm = squeeze(mean(xf, 1));
% zx = xm(:,[3 1]);
% [zx, ind] = sortrows(zx); 
% nx = length(ind)/nz;
% 
% xf = permute(xf, [1 3 2]);
% xf = xf(:,:,ind);
% nlf = permute(nlf, [1 3 2]);
% nlf = nlf(:,:,ind);
% uhf = permute(uhf, [1 3 2]);
% uhf = uhf(:,:,ind);
% udgf = permute(udgf, [1 3 2]);
% udgf = udgf(:,:,ind);
% 
% p1=porder+1;
% xa = squeeze(mean(reshape(xf, [p1 p1 3 nx nz]), [2,5]));
% x = squeeze(xa(:,1,:));
% y = squeeze(xa(:,2,:));
% 
% na = squeeze(mean(reshape(nlf, [p1 p1 3 nx nz]), [2,5]));
% n1 = squeeze(na(:,1,:));
% n2 = squeeze(na(:,2,:));
% n3 = squeeze(na(:,3,:));
% 
% ncu = size(uhf,2);
% uha = squeeze(mean(reshape(uhf, [p1 p1 ncu nx nz]), [2,5]));
% 
% nc = size(udgf,2);
% udga = squeeze(mean(reshape(udgf, [p1 p1 nc nx nz]), [2,5]));
% 
% xyMidChord = [0.0 0.0; 0.8 0.015; 1.0 0.0];
% lower = lowerSurfaceMask(x,y,1, xyMidChord);
% upper = find(sum(lower,1)==0);
% lower = find(sum(lower,1)>0);
% xl = x(:,lower);
% yl = y(:,lower);
% xu = x(:,upper);
% yu = y(:,upper);
% 
% figure(1); clf; 
% plot(xl, yl, '-b');  hold on; plot(xu, yu, '-r'); 
% axis tight; axis equal;
% 
% [p, txx, txy, txz, tyy, tyz, tzz, Qx, Qy, Qz] = nsfluxcart3d(uha(:,1:5,:), udga(:,6:end,:), 1.4, param(2), 0.72);
% p = squeeze(p); pl = p(:,lower); pu = p(:,upper);
% txx = squeeze(txx); 
% txy = squeeze(txy); 
% tyy = squeeze(tyy);
% 
% cp = -2*(squeeze(p)-pinf);
% t1 = squeeze(txx).*n1 + squeeze(txy).*n2;
% t2 = squeeze(txy).*n1 + squeeze(tyy).*n2;
% % Skin shear stress 
% cf = -2*(t1.*n2 - t2.*n1);
% %cf = 2*(txy.*(n2.^2-n1.^2) + txx.*n1.*n2 - tyy.*n1.*n2);
% 
% cfl = cf(:,lower);
% cfu = cf(:,upper);
% figure(1); clf; 
% plot(xu(2,:), cfu(2,:)); hold on;
% plot(xl(2,:), cfl(2,:)); axis tight;
% figure(2); clf; 
% plot(xu(2,:), cp(2,upper)); hold on;
% plot(xl(2,:), cp(2,lower)); axis tight;
% 
% [Cx,Cy] = surfaceForceCoefficients(cp,cf,x,y,master2d);
% CD =  Cx*cos(alpha) + Cy*sin(alpha);
% CL = -Cx*sin(alpha) + Cy*cos(alpha);
% 
% figure(1); clf; 
% plot(xu(2,:), txx(2,upper).*n1(2,upper)); hold on;
% plot(xl(2,:), txx(2,lower).*n1(2,lower)); axis tight;
% 
% 
% figure(1); clf; 
% plot(xu(2,:), txx(2,upper)); hold on;
% plot(xl(2,:), txx(2,lower)); axis tight;
% 
% figure(1); clf; 
% plot(xu(2,:), tyy(2,upper)); hold on;
% plot(xl(2,:), tyy(2,lower)); axis tight;
% 
% figure(1); clf; 
% plot(xu(2,:), txy(2,upper)); hold on;
% plot(xl(2,:), txy(2,lower)); axis tight;
% 
% figure(1); clf; 
% plot(xu(2,:), t1(2,upper)); hold on;
% plot(xl(2,:), t1(2,lower)); axis tight;
% 
% figure(1); clf; 
% plot(xu(2,:), t2(2,upper)); hold on;
% plot(xl(2,:), t2(2,lower)); axis tight;
% 
% figure(1); clf; 
% plot(xu(2,:), n1(2,upper)); hold on;
% plot(xl(2,:), n1(2,lower)); axis tight;
% 
% figure(1); clf; 
% plot(xu(2,:), n2(2,upper)); hold on;
% plot(xl(2,:), n2(2,lower)); axis tight;
% 
% figure(2); clf; plot(x, cf); axis tight;
% figure(3); clf; plot(x, squeeze(Qx), '.'); axis tight;
% 
% [xpe,telem,xpf,tface,perm] = masternodes(porder,3,1);
% 
% nref = 1;
% xmin = -0.05; xmax = 1.0;
% pres = eulereval3d(uhf(:,1:5,:),'p',gam,Minf)/pinf;
% cmin = min(pres(:)); cmax = max(pres(:));
% 
% figure(1);clf;
% plotudg(xf,pres,xpf,tface,porder,[],nref); colorbar;
% axis equal; axis tight; view(3);
% 
% figure(1);clf;
% plotudg(xf,pres,xpf,tface,porder,[],nref); colorbar; view(3);
% hold on;
% xt = xf; xt(:,3,:) = xt(:,3,:) + 0.03;
% plotudg(xt,pres,xpf,tface,porder,[cmin cmax],nref, xmin, xmax);
% xt(:,3,:) = xt(:,3,:) + 0.03;
% plotudg(xt,pres,xpf,tface,porder,[cmin cmax],nref, xmin, xmax);
% xt(:,3,:) = xt(:,3,:) + 0.03;
% plotudg(xt,pres,xpf,tface,porder,[cmin cmax],nref, xmin, xmax);
% xt(:,3,:) = xt(:,3,:) + 0.03;
% plotudg(xt,pres,xpf,tface,porder,[cmin cmax],nref, xmin, xmax);
% axis equal; axis tight; colormap('jet'); colorbar;
% set(gca,'FontSize',16);
% %exportgraphics(gca,"pressure16" + ".png",'Resolution',300);
% 
% % UDG = getsolution("dataout/outudg_t4000",dmd,n3);
% % UDGavg = getmeansolution("dataout/outreavg",dmd,n3);
% % [UDGspm,UDG2d] = spanwiseaverage(UDGavg,n1,nz,UDG,2,10);
% %
% % pde.visscalars = {"qcrit", 1, "u", 2, "p", 3};
% % pde.visvectors = {};
% % pde.paraview = "/Applications/ParaView-6.0.0.app/Contents/MacOS/paraview";
% %
% % qcrit = qcriterion(UDG);
% % ind = qcrit(:)>2e2;
% % qcrit(ind)=2e2;
% % ind = qcrit(:)<-200;
% % qcrit(ind)=-200;
% %
% % visfield = qcrit;
% % visfield(:,2,:) = UDG(:,2,:)./UDG(:,1,:);
% % visfield(:,3,:) = eulereval3d(UDG,'p',gamma,Ma);
% % vis(visfield,pde,mesh);
% %
% % figure(2); clf; scaplot(mesh2d, UDGspm(:,2,:)./UDGspm(:,1,:),[],2);
% % colormap('jet'); colorbar; axis([-0.05 1.35 -0.2 0.2]); set(gca,'FontSize',20);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average density');
% % exportgraphics(gca, "density.png",'Resolution',300);
% %
% % figure(3); clf; scaplot(mesh2d, eulereval3d(UDGspm,'p',gamma,Ma),[],2);
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average pressure');
% % exportgraphics(gca, "pressure.png",'Resolution',300);
% %
% % figure(4); clf; scaplot(mesh2d, eulereval3d(UDGspm,'vm',gamma,Ma),[],2);
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average velocity magnitude');
% % exportgraphics(gca, "velocity.png",'Resolution',300);
% %
% % figure(5); clf; scaplot(mesh2d, eulereval3d(UDG2d,'u',gamma,Ma),[-0.1 1.5],2,1);
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('instantaneous horizontal velocity');
% % exportgraphics(gca, "horizontalvelocity.png",'Resolution',300);
% %
% % figure(6); clf; scaplot(mesh2d, eulereval3d(UDG2d,'w',gamma,Ma),[-0.1 0.1],2);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('instantaneous spwanwise velocity');
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% % figure(7); clf; scaplot(mesh2d, eulereval3d(UDG2d,'vm',gamma,Ma),[0 1.5],2);
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('instantaneous velocity magnitude');
% %
% % figure(6); clf; scaplot(mesh2d, reavg(:,6,:), [],2);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average horizontal velocity');
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% %
% % figure(6); clf; scaplot(mesh2d, reavg(:,7,:), [],2);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average vertical velocity');
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% %
% % figure(6); clf; scaplot(mesh2d, reavg(:,8,:), [],2);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average spanwise velocity');
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% %
% % figure(6); clf; scaplot(mesh2d, reavg(:,9,:), [],2);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average pressure');
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% %
% % figure(6); clf; scaplot(mesh2d, reavg(:,17,:)-reavg(:,6,:).^2, [],2);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average uu');
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% %
% % figure(6); clf; scaplot(mesh2d, reavg(:,18,:)-reavg(:,7,:).^2, [],2);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average vv');
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% %
% % figure(6); clf; scaplot(mesh2d, reavg(:,19,:)-reavg(:,8,:).^2, [],2);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average ww');
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% %
% % figure(6); clf; scaplot(mesh2d, reavg(:,20,:)-reavg(:,6,:).*reavg(:,7,:), [],2);
% % xlabel('$x$','Interpreter','latex'); ylabel('$y$','Interpreter','latex'); title('time-average uv');
% % colormap('jet'); colorbar; axis([-0.05 1.5 -0.2 0.3]); set(gca,'FontSize',20);
% 
% 
% % wid = 1;              % adjust to the wall boundary marker
% % param = pde.physicsparam;
% % param(2) = 4.6e5;
% % xyMidChord = [0.0 0.0; 0.8 0.015; 1.0 0.0];
% % [Cp,Cf,x,Cp2d,Cf2d,x2d,Ch,Ch2d,stress] = getsurfacedata(master2d,mesh2d,udgavg,param,wid,1,1,xyMidChord);
% % figure(1);clf;plot(x(:,1),-Cf);axis tight;
% % 
% % figure(1);clf;plot(x(:,1),stress(:,1));axis tight;
