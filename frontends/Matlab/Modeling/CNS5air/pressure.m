function p = pressure(T, rho_i, Mw)
    RU = 8.314471468617452;
    p = T * sum(rho_i ./ Mw) * RU;
end
