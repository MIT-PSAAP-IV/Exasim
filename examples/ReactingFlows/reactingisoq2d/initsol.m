%load('solp2.mat')
srcdir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'frontends', 'Matlab');
addpath(char(srcdir + "/Modeling/CNS5air/"));

R = 288;
gam = 1.4;
gam1 = gam-1;
Minf = 7.6;
Re = 1.56*1e5;
T_ref  = 266.5;
rho_ref = 1.047e-3;
v_ref =  2500;
p_ref = R * rho_ref .* T_ref;

rho = sol(:,1,:);
v = sol(:,2:3,:)./sol(:,1,:);
ke = 0.5*(v(:,1,:).*v(:,1,:)+v(:,2,:).*v(:,2,:));
p = gam1*(sol(:,4,:) - rho.*ke);
T = gam*Minf^2 * p./rho;

[rho_phys, v_phys, T_phys, p_phys, e_phys, rhoE_phys] = computePhysicalStateFromNondim(sol, Minf, rho_ref, v_ref, T_ref);

figure(1); clf; scaplot(mesh, rho_phys,[],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

figure(2); clf; scaplot(mesh, v_phys(:,1,:),[0 v_ref],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

figure(3); clf; scaplot(mesh, T_phys(:,1,:),[T_ref 3000],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

figure(4); clf; scaplot(mesh, p_phys(:,1,:),[p_ref 5e3],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

v1 = v_phys(:,1,:);
v2 = v_phys(:,2,:);
[npe, nc, ne] = size(sol);
rho_species = zeros(5,npe*ne);
rhoE = zeros(npe,1,ne);
for i = 1:npe*ne
  [i p_phys(i), T_phys(i), [v1(i) v2(i)]]
  info = equilibrate(p_phys(i), T_phys(i), [v1(i) v2(i)]);
  rho_species(:,i) = info.rho_species(:);
  rho(i) = info.rho;
  rhoE(i) = info.rhoE;
end
rho_species = permute(reshape(rho_species, [5 npe ne]), [2 1 3]);

rhoE_phys = rhoE;
rhov_phys = v_phys(:,1,:).*rho;
rhov_phys(:,2,:) = v_phys(:,2,:).*rho;
%save initsol.mat sol mesh master pde rho_species rhov_phys rhoE_phys T_phys

figure(4); clf; scaplot(mesh, rho(:,1,:),[0 1e-2],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
%
figure(5); clf; scaplot(mesh, sol(:,1,:)*rho_ref,[0 1e-2],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

figure(6); clf; scaplot(mesh, rhoE,[0 2.5e5],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

rhoE_phys = p_phys/gam1 + 0.5*(sol(:,1,:)*rho_ref).*(v_phys(:,1,:).*v_phys(:,1,:) + v_phys(:,2,:).*v_phys(:,2,:));
figure(7); clf; scaplot(mesh, rhoE_phys,[],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

figure(8); clf; scaplot(mesh, sol(:,4,:),[],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);

for i = 1:5
figure(i); clf; scaplot(mesh, rho_species(:,i,:)./rho,[],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
end

figure(7); clf; scaplot(mesh, T_phys,[],1);
colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
