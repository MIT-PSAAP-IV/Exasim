function checkconformalinterface(app)
    # Preserve matching inputs; reject features absent from this release.
    if app isa AbstractVector
        foreach(checkconformalinterface, app)
        return
    end
    value(key, default) = app isa AbstractDict ? get(app, key, get(app, String(key), default)) :
        (hasproperty(app, key) ? getproperty(app, key) : default)
    if value(:interfacetype, 0) != 0 || !isempty(value(:interactiontable, [])) ||
            !isempty(value(:interactions, [])) || !isempty(value(:interactionfile, ""))
        error("Nonmatching interfaces and generalized interactions are not available in this source release.")
    end
end
