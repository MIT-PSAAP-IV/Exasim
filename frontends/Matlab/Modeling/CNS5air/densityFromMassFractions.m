function rho = densityFromMassFractions(T, P, Y, Mw)
%DENSITYFROMMASSFRACTIONS Compute mixture density from mass fractions.
%
%   rho = densityFromMassFractions(T, P, Y, Mw)
%
% Inputs:
%   T  : temperature [K] (scalar)
%   P  : pressure [Pa] (scalar)
%   Y  : mass fraction vector (ns x 1 or 1 x ns)
%   Mw : molecular weight vector [kg/mol] (ns x 1 or 1 x ns)
%
% Output:
%   rho : mixture density [kg/m^3]
%
% Formula:
%   rho = P / (RU * T * sum(Y ./ Mw))
%
% Notes:
%   - Y must sum to 1
%   - Mw must be in kg/mol

    RU = 8.314471468617452;   % universal gas constant [J/(mol K)]

    % Ensure column vectors
    Y  = Y(:);
    Mw = Mw(:);

    % Compute density
    rho = P ./ (RU .* T .* sum(Y ./ Mw));

end