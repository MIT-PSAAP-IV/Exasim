function [p, u, rhoE, e] = idealGas_pe_from_rhoRhouT(rho, rhou, T, gamma, R)
%idealGas_pe_from_rhoRhouT  Compute p, u, rhoE, e from (rho, rho*u, T) for an ideal gas.
%
% Inputs:
%   rho   : density (scalar)
%   rhou  : momentum vector (2x1 or 3x1) [rho*u; rho*v; (rho*w)]
%   T     : temperature (K)
%   gamma : ratio of specific heats
%   R     : specific gas constant (J/kg/K)
%
% Outputs:
%   p     : pressure (Pa)
%   u     : velocity vector
%   rhoE  : total energy density (rho*E)
%   e     : internal specific energy (J/kg)
%
% Relations:
%   p = rho*R*T
%   e = p / ((gamma-1)*rho) = R*T/(gamma-1)
%   E = e + 0.5*|u|^2
%   rhoE = rho*E

    % --- checks ---
    if rho <= 0
        error('rho must be positive.');
    end
    if T <= 0
        error('T must be positive.');
    end
    if gamma <= 1
        error('gamma must be > 1.');
    end
    if R <= 0
        error('R must be positive.');
    end

    rhou = rhou(:);
    u = rhou / rho;

    % --- EOS ---
    p = rho * R * T;

    % --- internal energy ---
    e = p / ((gamma - 1) * rho);  % = R*T/(gamma-1)

    % --- total energy density ---
    ke = 0.5 * dot(u,u);
    rhoE = rho * (e + ke);
end