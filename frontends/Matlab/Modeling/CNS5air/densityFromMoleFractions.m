function rho = densityFromMoleFractions(T, P, X, Mw)

    RU = 8.314471468617452;   % J/(mol K)

    X  = X(:);
    Mw = Mw(:);

    % mixture molecular weight
    Wmix = sum(X .* Mw);

    % ideal gas EOS
    rho = P .* Wmix ./ (RU .* T);

end