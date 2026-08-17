function mesh = mkmesh_cyl(porder,a,b,c,alpha,m,n)

if nargin < 6
  m = 60;
end
if nargin < 7
  n= 40;
end

mesh = mkmesh_square(m,n,porder,1,1,1,1,1);
mesh.p(1,:) = logdec(mesh.p(1,:), alpha);
mesh.dgnodes(:,1,:) = logdec(mesh.dgnodes(:,1,:), alpha);
mesh = mkmesh_halfcircle(mesh, a, b, c, pi/2, 3*pi/2);
mesh.porder = porder;
mesh.boundaryexpr = {@(p) sqrt(p(1,:).^2+p(2,:).^2)<a+1e-6, @(p) p(1,:)>-1e-7, @(p) abs(p(1,:))<20};
mesh.periodicexpr = {};

mesh.xpe   = mesh.plocal;
mesh.telem = mesh.tlocal;
