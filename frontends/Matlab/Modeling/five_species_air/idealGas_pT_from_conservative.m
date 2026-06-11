function [p, T, u, E, e] = idealGas_pT_from_conservative(rho, rhou, rhoE, gamma, R)
%IDEALGAS_PT_FROM_CONSERVATIVE  Compute p and T from (rho, rho*u, rhoE) for an ideal gas.
%
% Inputs:
%   rho   : density (scalar)
%   rhou  : momentum vector (2x1 or 3x1) [rho*u; rho*v; (rho*w)]
%   rhoE  : total energy density (rho*E)
%   gamma : ratio of specific heats
%   R     : specific gas constant (J/kg/K), e.g. air: 287.0529
%
% Outputs:
%   p : pressure (Pa)
%   T : temperature (K)
%   u : velocity vector
%   E : total specific energy
%   e : internal specific energy
%
% Notes:
%   E = e + 0.5*|u|^2
%   p = (gamma-1)*rho*e
%   T = p/(rho*R)

    % basic checks
    if rho <= 0
        error('rho must be positive.');
    end
    if gamma <= 1
        error('gamma must be > 1.');
    end
    if R <= 0
        error('R must be positive.');
    end

    rhou = rhou(:);
    u = rhou / rho;

    % energies
    E = rhoE / rho;
    ke = 0.5 * dot(u,u);
    e  = E - ke;

    if e <= 0
        error('Internal energy is non-positive (e = %g). Check rhoE/rhou.', e);
    end

    % ideal gas EOS / caloric relation
    p = (gamma - 1) * rho * e;
    T = p / (rho * R);

    if T <= 0
        error('Computed temperature is non-positive (T = %g).', T);
    end
end