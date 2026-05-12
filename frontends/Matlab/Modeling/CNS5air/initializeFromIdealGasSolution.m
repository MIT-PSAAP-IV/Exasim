function [rho_species, rhov_phys, rhoE_phys, rho_phys, T_phys, p_phys] = initializeFromIdealGasSolution(sol, Minf, rho_ref, v_ref, T_ref)

[rho_phys, v_phys, T_phys, p_phys] = computePhysicalStateFromNondim(sol, Minf, rho_ref, v_ref, T_ref);

v1 = v_phys(:,1,:);
v2 = v_phys(:,2,:);
[npe, ~, ne] = size(sol);
rho_species = zeros(5,npe*ne);
rhoE_phys = zeros(npe,1,ne);
for i = 1:npe*ne  
  info = equilibrate(p_phys(i), T_phys(i), [v1(i) v2(i)]);
  rho_species(:,i) = info.rho_species(:);
  rho_phys(i) = info.rho;
  rhoE_phys(i) = info.rhoE;
end
rho_species = permute(reshape(rho_species, [5 npe ne]), [2 1 3]);
rhov_phys = rho_phys .* v_phys;




% figure(1); clf; scaplot(mesh, rho_phys,[],1);
% colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
% hold on; plot(mesh.dgnodes(:,1,1),mesh.dgnodes(:,2,1),'o');
% 
% figure(2); clf; scaplot(mesh, v_phys(:,1,:),[0 v_ref],1);
% colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
% 
% figure(3); clf; scaplot(mesh, T_phys(:,1,:),[T_ref 15000],1);
% colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
% hold on; plot(mesh.dgnodes(:,1,1),mesh.dgnodes(:,2,1),'o');
% 
% figure(4); clf; scaplot(mesh, p_phys(:,1,:),[p_ref 1e5],1);
% colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
% hold on; plot(mesh.dgnodes(:,1,1),mesh.dgnodes(:,2,1),'o');
% 
% figure(4); clf; scaplot(mesh, rho(:,1,:),[0 1e-2],1);
% colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
% % 
% figure(5); clf; scaplot(mesh, sol(:,1,:)*rho_ref,[0 1e-2],1);
% colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
% 
% figure(6); clf; scaplot(mesh, rhoE_phys,[0 2.5e5],1);
% colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
% 
% rhoE_phys_phys = p_phys/gam1 + 0.5*(sol(:,1,:)*rho_ref).*(v_phys(:,1,:).*v_phys(:,1,:) + v_phys(:,2,:).*v_phys(:,2,:));
% figure(7); clf; scaplot(mesh, rhoE_phys_phys,[],1);
% colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
% 
% figure(8); clf; scaplot(mesh, sol(:,4,:),[],1);
% colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
% 
% figure(7); clf; scaplot(mesh, rho_species(:,5,:)./rho,[],1);
% colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
% 
% figure(7); clf; scaplot(mesh, T_phys,[],1);
% colorbar; colormap('jet'); axis on; axis equal; axis tight; set(gca,'FontSize',16);
