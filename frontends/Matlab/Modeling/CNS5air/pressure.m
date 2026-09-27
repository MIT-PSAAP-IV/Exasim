function p = pressure(T, rho_i, Mw)

    if nargin < 3
        [~, Mw, ~] = thermodynamicsModels();
    end

    RU = 8.314471468617452;
    p = T .* sum(rho_i ./ Mw) * RU;

end
