function [rho_ref, v_ref, rhoe_ref, p_ref, T_ref, mu_ref, kappa_ref, lambda_ref, cp_ref, cv_ref] = getReferenceState(p_phys, T_phys, v_phys)

info = equilibrate(p_phys, T_phys, v_phys);
rho_species_ref = info.rho_species;
rho_ref = sum(rho_species_ref);
v_ref = v_phys;
T_ref = T_phys;
rhoe_ref = rho_ref*v_ref^2;
p_ref = rho_ref*v_ref^2;

[~, ~, ~, ~, mu_ref, kappa_ref, lambda_ref, cp, cv] = transportcoefficients(T_ref, rho_species_ref(:), 1e4);
mu_ref = double(mu_ref); 
kappa_ref = double(kappa_ref);
lambda_ref = double(lambda_ref);
cp = double(cp);
cv = double(cv);
cp_ref = sum(cp.*rho_species_ref(:)/rho_ref);
cv_ref = sum(cv.*rho_species_ref(:)/rho_ref);

