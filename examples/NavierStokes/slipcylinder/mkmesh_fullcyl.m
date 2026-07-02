function mesh = mkmesh_fullcyl(porder, R)

if nargin < 2
  R = 1;
end

nr = 81;
nt = 120;
a = R;
b = 4.5*a;
c = 5.0*a;

rho1d = logdec(linspace(0,1,nr)',4);
theta1d = linspace(3*pi/2,-pi/2,nt+1);
theta1d(end) = [];

[rho,theta] = ndgrid(rho1d,theta1d);
d = -b*cos(theta) + c*(1+cos(theta));
r = d + rho.*(a-d);

p = [r(:).*cos(theta(:)), r(:).*sin(theta(:))];
[~, imax] = max(p(:,2));
x_split = p(imax,1);
x_tol = 1e-8*max(1,abs(x_split));

t = zeros((nr-1)*nt,4);
e = 1;
for j = 1:nt
    jp = mod(j,nt) + 1;
    for i = 1:(nr-1)
        n1 = i + (j-1)*nr;
        n2 = i + 1 + (j-1)*nr;
        n3 = i + 1 + (jp-1)*nr;
        n4 = i + (jp-1)*nr;
        t(e,:) = [n1 n2 n3 n4];
        e = e + 1;
    end
end

outerexpr = ['all(sqrt(sum(p.^2,2))>' num2str(b-1e-6,17) ')'];
bndexpr = {['all(sqrt(sum(p.^2,2))<' num2str(a+1e-6,17) ')'], ...
           [outerexpr ' && mean(p(:,1))<=' num2str(x_split+x_tol,17)], ...
           [outerexpr ' && mean(p(:,1))>' num2str(x_split+x_tol,17)]};
mesh = mkmesh(p,t,porder,bndexpr,1,1);
mesh.p = p';
mesh.t = t';
mesh.porder = porder;
mesh.boundaryexpr = {@(p) sqrt(p(1,:).^2+p(2,:).^2)<R+1e-6, ...
                     @(p) sqrt(p(1,:).^2+p(2,:).^2)>b-1e-6 & p(1,:)<=x_split+x_tol, ...
                     @(p) sqrt(p(1,:).^2+p(2,:).^2)>b-1e-6 & p(1,:)>=x_split-x_tol};
mesh.periodicexpr = {};
mesh.x_split = x_split;
mesh.fcurved = true(size(mesh.f,1),1);
mesh.tcurved = true(size(mesh.t,2),1);

xi = mesh.plocal(:,1);
eta = mesh.plocal(:,2);
npe = size(mesh.plocal,1);
ne = size(t,1);
mesh.dgnodes = zeros(npe,2,ne);
e = 1;
for j = 1:nt
    theta1 = theta1d(j);
    theta2 = theta1 - 2*pi/nt;
    thetae = theta1 + eta*(theta2-theta1);
    for i = 1:(nr-1)
        rhoe = rho1d(i) + xi*(rho1d(i+1)-rho1d(i));
        de = -b*cos(thetae) + c*(1+cos(thetae));
        re = de + rhoe.*(a-de);
        mesh.dgnodes(:,1,e) = re.*cos(thetae);
        mesh.dgnodes(:,2,e) = re.*sin(thetae);
        e = e + 1;
    end
end

figure(1); clf; meshplot(mesh);

mesh.f = facenumbering(mesh.p, mesh.t, mesh.elemtype, ...
                       mesh.boundaryexpr, mesh.periodicexpr);
mesh.xpe   = mesh.plocal;
mesh.telem = mesh.tlocal;

nb = max(mesh.f(:));
colors = lines(nb);
figure(2); clf; boundaryplot(mesh,1,colors(1,:));
hold on;
for i = 2:nb
  boundaryplot(mesh,i,colors(i,:));
end
axis equal; axis tight;