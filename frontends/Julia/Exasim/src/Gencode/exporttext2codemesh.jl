function exporttext2codemesh(mesh, dest, suffix="")
    # Export Text2Code mesh-related binary files with the same layout used by
    # the Matlab exporttext2codemesh helper.
    dest = abspath(String(dest))
    suffix = String(suffix)
    mkpath(dest)

    p = _t2c_get(mesh, :p)
    t = _t2c_get(mesh, :t)
    (isnothing(p) || _t2c_empty(p)) && error("exporttext2codemesh: mesh must contain a nonempty p array.")
    (isnothing(t) || _t2c_empty(t)) && error("exporttext2codemesh: mesh must contain a nonempty t array.")

    _t2c_writebin(joinpath(dest, "grid$(suffix).bin"), vcat(collect(size(p)), collect(size(t)), vec(p), vec(t)))

    optional = [
        (:dgnodes, "xdg"),
        (:udg, "udg"),
        (:vdg, "vdg"),
        (:wdg, "wdg"),
    ]
    for (meshkey, filenamebase) in optional
        value = _t2c_get(mesh, meshkey)
        if !isnothing(value) && !_t2c_empty(value)
            _t2c_writebin(joinpath(dest, "$(filenamebase)$(suffix).bin"), vcat(collect(size(value)), vec(value)))
        end
    end

    return dest
end
