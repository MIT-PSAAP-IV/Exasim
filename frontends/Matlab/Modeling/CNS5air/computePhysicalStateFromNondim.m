function [rho_phys, v_phys, T_phys, p_phys, e_phys, rhoE_phys] = computePhysicalStateFromNondim(sol, Minf, rho_ref, v_ref, T_ref)

R = 287;
gam = 1.4;
gam1 = gam-1;
rho = sol(:,1,:);
v = sol(:,2:3,:)./sol(:,1,:);
ke = 0.5*(v(:,1,:).*v(:,1,:)+v(:,2,:).*v(:,2,:));
p = gam1*(sol(:,4,:) - rho.*ke);
T = gam*Minf^2 * p./rho;

rho_phys = rho*rho_ref;
v_phys = v*v_ref;
T_phys = T*T_ref;
p_phys = R * rho_phys .* T_phys;
e_phys = p_phys ./ ((gam - 1) * rho_phys); 

% --- total energy density ---
ke = 0;
for i = 1:size(v_phys,2)
  ke = ke + v_phys(:,i,:) .* v_phys(:,i,:);
end
ke = 0.5 * ke;
rhoE_phys = rho_phys .* (e_phys + ke);

