
srcdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab');
addpath(char(srcdir + "/Mesh/adaptivity/"));

qmin = 0.2; qmax = 0.8; diffcoeff = 1e-3;
bcs = [3;3;3;2];
params = [1 1 1];
ndg2cg = 30;

mesh = mkmesh_isoq2d(2, 5e-4);
[~,cgelcon,rowent2elem,colent2elem,cgent2dgent] = mkcgent2dgent(mesh.dgnodes,1e-8);

avField = computeavfield2dchem(mesh.dgnodes, udg, 8, 40);
avField = dg2cg2(avField, cgelcon, colent2elem, rowent2elem);
avField = dg2cg2(avField, cgelcon, colent2elem, rowent2elem);
avField = dg2cg2(avField, cgelcon, colent2elem, rowent2elem);
dist = meshdist3(mesh.f,mesh.dgnodes,master.perm,4); % distance to the wall
avField = avField.*tanh(dist*40);
eta0 = avField/max(avField(:));

mesh0 = mesh;
eta = eta0;
for n = 1:8
  [solle,mu,lambda,fx,fy,h] = meshadapt2d(mesh, master, cgelcon, colent2elem, rowent2elem, cgent2dgent, eta, qmin, qmax, diffcoeff, bcs, params, ndg2cg);
  mesh.dgnodes = mesh.dgnodes + 0.2*solle(:,1:2,:);
  eta = fieldatdgnodes(mesh0, master, eta0, mesh.dgnodes);
end

figure(1); clf; scaplot(mesh,eta(:,1,:)); axis on; axis equal; axis tight; colorbar;
figure(2); clf; scaplot(mesh,h(:,1,:)); axis on; axis equal; axis tight; colorbar;
figure(3); clf; scaplot(mesh,mu(:,1,:)); axis on; axis equal; axis tight; colorbar;
figure(4); clf; scaplot(mesh,lambda(:,1,:)); axis on; axis equal; axis tight; colorbar;
figure(5); clf; scaplot(mesh,fx(:,1,:)); axis on; axis equal; axis tight; colorbar;
figure(6); clf; scaplot(mesh,fy(:,1,:)); axis on; axis equal; axis tight; colorbar;

figure(7); clf; scaplot(mesh,solle(:,1,:)); axis on; axis equal; axis tight; colorbar;
figure(8); clf; scaplot(mesh,solle(:,2,:)); axis on; axis equal; axis tight; colorbar;
figure(9); clf; meshplot(mesh,1); axis on; axis equal; axis tight;

udg1 = fieldatdgnodes(mesh0, master, udg, mesh.dgnodes);
wdg1 = fieldatdgnodes(mesh0, master, wdg, mesh.dgnodes);
figure(10); clf; scaplot(mesh0, wdg(:,1,:,end), [], 1); colorbar; colormap('jet');
figure(11); clf; scaplot(mesh, wdg1(:,1,:,end), [], 1); colorbar; colormap('jet');
