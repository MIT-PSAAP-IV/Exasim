function mesh = mkmesh_cyl(porder)
%MKMESH_CYL Mesh used by the Mach-8 hypersonic cylinder examples.
%
% This is the same geometry/topology construction used by
% examples/NavierStokes/hypersoniccylinder_mach8 so the equilibrium-air case
% can be compared directly with the ideal-gas case.

mesh = mkmesh_square(31,21,porder,1,1,1,1,1);
mesh.p(1,:) = logdec(mesh.p(1,:), 6);
mesh.dgnodes(:,1,:) = logdec(mesh.dgnodes(:,1,:), 6);
mesh = mkmesh_halfcircle(mesh, 1, 3, 4.7, pi/2, 3*pi/2);
mesh.porder = porder;
mesh.boundaryexpr = {@(p) sqrt(p(1,:).^2+p(2,:).^2)<1+1e-6, ...
                     @(p) p(1,:)>-1e-7, ...
                     @(p) abs(p(1,:))<20};
mesh.periodicexpr = {};
end
