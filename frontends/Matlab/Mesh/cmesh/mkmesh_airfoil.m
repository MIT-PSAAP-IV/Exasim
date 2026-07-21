function mesh = mkmesh_airfoil(xf, yf, porder, nxw, nflr, nflf, nfuf, nfur, nr)
%   nxw  : number of subdivison in the wake
%   nflr : number of subdivision in the lower foil (rear)
%   nflf : number of subdivision in the lower foil (front)
%   nfuf : number of subdivisions in the upper foil (front)
%   nfur : number of subdivisions in the upper foil (rear)
%   nr   : number of subdivisions in the radial direction

elemtype = 1;
nodetype = 1;

TEC = 15;
[x,y] = cmeshparam6(nxw, nflr, nflf, nfuf, nfur, nr, ...
                    [TEC, 1, 1, 1, 1, TEC, 1, 1, 1, 1, TEC], ...
                    [10, 10, 10, 10, 10, 10, 10]*25);

Rx = 10; Ry = 10;
[xm, ym] = cmeshmap(xf, yf, x, y, Rx, Ry);
% fix the wake gap
xm(1,1:nxw) = xm(1,end:-1:end-(nxw-1));
ym(1,1:nxw) = ym(1,end:-1:end-(nxw-1));

bndexpr={'sqrt((p(:,1)-.5).^2+p(:,2).^2)<2','true'};
[p,t] = cart2dg(elemtype,1,xm,ym);

figure(1); clf; simpplot(p,t); %pause

mesh = mkmesh(p,t,porder,bndexpr,elemtype,nodetype);
mesh.p = mesh.p';
mesh.t = mesh.t';
mesh.boundaryexpr = {@(p) sqrt((p(1,:)-.5).^2+p(2,:).^2)<3, ...
                     @(p) abs(p(1,:))< 1e6 + 1e-6};
mesh.boundarycondition = [1;2];

% The C-grid mapping already supplies curved high-order DG nodes. Do not
% re-project the Eppler boundary with an approximate implicit distance.
mesh.curvedboundary = [0 0];
mesh.curvedboundaryexpr = {@(p) 0*p(1,:), @(p) 0*p(1,:)};
mesh.periodicexpr = {};
mesh.f = facenumbering(mesh.p,mesh.t,mesh.elemtype, mesh.boundaryexpr,mesh.periodicexpr);

figure(2); clf; meshplot(mesh);
