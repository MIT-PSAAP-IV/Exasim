function [rho_species_phys, rho_phys, rhov_phys, rhoE_phys, mu_phys, kappa_phys, lambda_phys, cp_phys, cv_phys] = getEquilibriumState(p_phys, T_phys, v_phys)

alpha = 1e4;

info = equilibrate(p_phys, T_phys, v_phys);
rho_species_phys = info.rho_species;
rho_phys = sum(rho_species_phys);
rhov_phys = rho_phys*v_phys;

[rhoE_phys, p1_phys, ~] = energyFromSpecies(rho_species_phys, T_phys, v_phys, alpha);

if abs(p_phys-p1_phys)/p1_phys>1e-4  
  error("Pressure is wrong");
end

[~, ~, ~, ~, mu_phys, kappa_phys, lambda_phys, cp, cv] = transportcoefficients(T_phys, rho_species_phys(:), alpha);
cp_phys = sum(cp.*rho_species_phys(:)/rho_phys);
cv_phys = sum(cv.*rho_species_phys(:)/rho_phys);
