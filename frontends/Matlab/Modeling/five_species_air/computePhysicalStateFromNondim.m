function [rho_phys, v_phys, T_phys, p_phys, e_phys, rhoE_phys] = computePhysicalStateFromNondim(rho, v, T, rho_ref, v_ref, T_ref)

R = 287;
gamma = 1.4;

rho_phys = rho*rho_ref;
v_phys = v*v_ref;
T_phys = T*T_ref;
p_phys = R * rho_phys .* T_phys;
e_phys = p_phys ./ ((gamma - 1) * rho_phys); 

% --- total energy density ---
ke = 0;
for i = 1:size(v_phys,2)
  ke = ke + v_phys(:,i,:) .* v_phys(:,i,:);
end
ke = 0.5 * ke;
rhoE_phys = rho_phys .* (e_phys + ke);

