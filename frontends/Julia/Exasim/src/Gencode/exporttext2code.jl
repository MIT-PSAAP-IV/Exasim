function exporttext2code(pde, mesh, dest="")
    if isempty(dest)
        if hasproperty(pde, :exporttext2code) && !isempty(getproperty(pde, :exporttext2code))
            dest = getproperty(pde, :exporttext2code)
        elseif hasproperty(pde, :exportt2c) && !isempty(getproperty(pde, :exportt2c))
            dest = getproperty(pde, :exportt2c)
        else
            error("exporttext2code: no destination (pass dest or set pde.exporttext2code).")
        end
    end

    dest = abspath(dest)
    mkpath(dest)

    _t2c_infer_dimensions!(pde, mesh)

    println("Export Exasim Text2Code package to $dest ...")
    genpdemodel(pde, joinpath(dest, "pdemodel.txt"))

    files = _t2c_write_binaries(mesh, dest)
    _t2c_write_pdeapp(pde, mesh, files, joinpath(dest, "pdeapp.txt"))
    _t2c_write_readme(dest)

    println("Exported Text2Code package: $dest")
    return dest
end

function _t2c_pde_dict(pde)
    if pde isa AbstractDict
        return Dict{String,Any}(string(k) => v for (k, v) in pde)
    end
    app = Dict{String,Any}()
    for name in fieldnames(typeof(pde))
        app[string(name)] = getfield(pde, name)
    end
    return app
end

function _t2c_infer_dimensions!(app, mesh)
    p = _t2c_get(mesh, :p)
    if !isnothing(p)
        nd = size(p, 1)
        _t2c_setapp!(app, "nd", nd)
        _t2c_setapp!(app, "ncx", nd)
    else
        nd = Int(_t2c_getapp(app, "nd", 1))
    end
    ncu = Int(_t2c_getapp(app, "ncu", 1))
    model = lowercase(String(_t2c_getapp(app, "model", _t2c_getapp(app, "pdemodel", "ModelD"))))
    nc = model == "modelc" ? ncu : ncu * (nd + 1)
    nc = max(Int(_t2c_getapp(app, "nc", nc)), nc)
    _t2c_setapp!(app, "nc", nc)
    _t2c_setapp!(app, "ncq", max(nc - ncu, 0))

    vdg = _t2c_get(mesh, :vdg)
    if !isnothing(vdg)
        _t2c_setapp!(app, "nco", size(vdg, 2))
    end
    wdg = _t2c_get(mesh, :wdg)
    if !isnothing(wdg)
        _t2c_setapp!(app, "ncw", size(wdg, 2))
    end
    return app
end

function _t2c_getapp(app, key, default)
    app isa AbstractDict && return get(app, key, get(app, Symbol(key), default))
    sym = Symbol(key)
    return hasproperty(app, sym) ? getproperty(app, sym) : default
end

function _t2c_setapp!(app, key, value)
    if app isa AbstractDict
        app[key] = value
    else
        setproperty!(app, Symbol(key), value)
    end
end

function _t2c_write_binaries(mesh, dest)
    files = Dict{String,String}("meshfile" => "grid.bin")
    p = _t2c_get(mesh, :p)
    t = _t2c_get(mesh, :t)
    isnothing(p) && error("exporttext2code: mesh must contain p array.")
    isnothing(t) && error("exporttext2code: mesh must contain t array.")
    _t2c_writebin(joinpath(dest, "grid.bin"), vcat(collect(size(p)), collect(size(t)), vec(p), vec(t)))

    optional = [
        (:dgnodes, "xdgfile", "xdg.bin"),
        (:udg, "udgfile", "udg.bin"),
        (:vdg, "vdgfile", "vdg.bin"),
        (:wdg, "wdgfile", "wdg.bin"),
    ]
    for (meshkey, appkey, filename) in optional
        value = _t2c_get(mesh, meshkey)
        if !isnothing(value)
            _t2c_writebin(joinpath(dest, filename), vcat(collect(size(value)), vec(value)))
            files[appkey] = filename
        end
    end
    return files
end

function _t2c_write_pdeapp(pde, mesh, files, path)
    app = _t2c_pde_dict(pde)
    app["modelfile"] = "pdemodel.txt"
    app["meshfile"] = files["meshfile"]
    for key in ["xdgfile", "udgfile", "vdgfile", "wdgfile"]
        if haskey(files, key)
            app[key] = files[key]
        else
            pop!(app, key, nothing)
        end
    end

    app["discretization"] = Int(app["hybrid"]) == 1 ? "hdg" : "ldg"
    app["NewtonIter"] = get(app, "NLiter", get(app, "NewtonIter", 20))
    app["NewtonTol"] = get(app, "NLtol", get(app, "NewtonTol", 1e-6))
    app["GMRESiter"] = get(app, "linearsolveriter", get(app, "GMRESiter", 200))
    app["GMREStol"] = get(app, "linearsolvertol", get(app, "GMREStol", 1e-3))
    app["ncv"] = get(app, "nco", get(app, "ncv", 0))
    app["frontendgenerated"] = 0
    if haskey(app, "physicsparamsweep") && !_t2c_empty(app["physicsparamsweep"])
        app["physicsparamcases"] = _t2c_normalize_sweep_cases(app["physicsparamsweep"], length(vec(app["physicsparam"])))
    end

    boundaryconditions = _t2c_get(mesh, :boundarycondition)
    if !isnothing(boundaryconditions)
        app["boundaryconditions"] = boundaryconditions
    end
    app["boundaryexpressions"] = _t2c_string_list(_t2c_get(mesh, :boundaryexpr), "boundaryexpr")

    curved = _t2c_get(mesh, :curvedboundary)
    app["curvedboundaries"] = isnothing(curved) ? [] : curved
    curvedexpr = _t2c_get(mesh, :curvedboundaryexprs)
    if isnothing(curvedexpr)
        curvedexpr = _t2c_get(mesh, :curvedboundaryexpr)
    end
    if isnothing(curvedexpr)
        curvedexpr = fill("", length(vec(app["boundaryconditions"])))
    end
    app["curvedboundaryexprs"] = _t2c_string_list(curvedexpr, "curvedboundaryexpr")

    _t2c_add_periodic!(app, mesh)
    app["interfaceconditions"] = something(_t2c_get(mesh, :interfacecondition), [])

    keys = [
        "model", "modelfile", "meshfile", "xdgfile", "udgfile", "vdgfile", "wdgfile",
        "discretization", "platform", "mpiprocs", "debugmode", "runmode", "modelnumber",
        "builtinmodelID", "frontendgenerated",
        "nodetype", "ncu", "ncv", "ncw", "neb", "nfb", "linearproblem", "subproblem",
        "saveParaview", "physicsparamwarmstart", "tdep", "wave", "porder", "pgauss",
        "temporalscheme", "torder", "nstage", "convStabMethod", "diffStabMethod",
        "rotatingFrame", "viscosityModel", "SGSmodel", "ALE", "AV", "AVsmoothingIter",
        "frozenAVflag", "nonlinearsolver", "linearsolver", "NewtonIter", "NewtonTol",
        "GMRESiter", "GMRESrestart", "GMREStol", "GMRESortho", "ppdegree", "RBdim",
        "matvecorder", "matvectol", "precMatrixType", "preconditioner", "time",
        "NLparam", "tau", "dt", "dae_alpha", "dae_beta", "dae_gamma", "dae_epsilon",
        "dae_steps", "dae_dt", "physicsparam", "physicsparamcases", "externalparam",
        "vindx", "avparam1", "avparam2", "stgib", "stgdata", "stgparam",
        "boundaryconditions", "boundaryexpressions", "curvedboundaries",
        "curvedboundaryexprs", "periodicboundaries1", "periodicexprs1",
        "periodicboundaries2", "periodicexprs2", "interfaceconditions",
        "interfacefluxmap", "wmModelIDs", "wmBoundaries", "wmDistances",
        "saveSolFreq", "saveSolOpt", "timestepOffset", "saveSolBouFreq", "ibs",
        "compudgavg", "extFhat", "extUhat", "extStab", "saveResNorm",
    ]

    open(path, "w") do io
        for key in keys
            if haskey(app, key) && !_t2c_empty(app[key])
                println(io, key * " = " * _t2c_format_value(app[key], key) * ";")
            end
        end
    end
end

function _t2c_add_periodic!(app, mesh)
    periodic = _t2c_get(mesh, :periodicboundary)
    if isnothing(periodic) || length(periodic) == 0
        app["periodicboundaries1"] = []
        app["periodicboundaries2"] = []
        app["periodicexprs1"] = []
        app["periodicexprs2"] = []
        return
    end

    b1 = Int[]
    b2 = Int[]
    e1 = String[]
    e2 = String[]
    for row in eachrow(periodic)
        push!(b1, row[1])
        append!(e1, _t2c_periodic_expr(row[2]))
        push!(b2, row[3])
        append!(e2, _t2c_periodic_expr(row[4]))
    end
    app["periodicboundaries1"] = b1
    app["periodicboundaries2"] = b2
    app["periodicexprs1"] = e1
    app["periodicexprs2"] = e2
end

function _t2c_periodic_expr(expr)
    values = _t2c_string_list([expr], "periodic expression")
    values[1] == "xy" && return ["x", "y"]
    values[1] == "xz" && return ["x", "z"]
    values[1] == "yz" && return ["y", "z"]
    return values
end

function _t2c_get(obj, key::Symbol)
    if obj isa AbstractDict
        return get(obj, key, get(obj, string(key), nothing))
    end
    return hasproperty(obj, key) ? getproperty(obj, key) : nothing
end

function _t2c_writebin(filename, data)
    open(filename, "w") do io
        write(io, Float64.(data))
    end
end

function _t2c_string_list(value, label)
    isnothing(value) && return String[]
    value isa AbstractString && return [String(value)]
    out = String[]
    for item in vec(value)
        if !(item isa AbstractString)
            error("exporttext2code: $label entries must be strings for Text2Code export.")
        end
        push!(out, String(item))
    end
    return out
end

function _t2c_normalize_sweep_cases(spec, nparam)
    if spec isa AbstractDict
        haskey(spec, :samples) && return _t2c_normalize_sweep_cases(spec[:samples], nparam)
        haskey(spec, "samples") && return _t2c_normalize_sweep_cases(spec["samples"], nparam)
        haskey(spec, :values) && return _t2c_normalize_sweep_cases(spec[:values], nparam)
        haskey(spec, "values") && return _t2c_normalize_sweep_cases(spec["values"], nparam)
        grid = haskey(spec, :grid) ? spec[:grid] : (haskey(spec, "grid") ? spec["grid"] : nothing)
        if !isnothing(grid)
            length(grid) == nparam || error("physicsparamsweep grid must have one value vector per physics parameter.")
            meshgrids = Iterators.product((vec(g) for g in grid)...)
            cases = zeros(Float64, length(collect(meshgrids)), nparam)
            for (i, point) in enumerate(Iterators.product((vec(g) for g in grid)...))
                cases[i, :] .= collect(point)
            end
            return cases
        end
        error("physicsparamsweep dict must contain samples, values, or grid.")
    end
    arr = Float64.(spec)
    if ndims(arr) == 1
        arr = nparam == 1 ? reshape(arr, :, 1) : reshape(arr, 1, :)
    end
    size(arr, 2) == nparam || error("Each physicsparamsweep row must contain $nparam physics parameters.")
    all(isfinite, arr) || error("physicsparamsweep cases must contain finite numeric values.")
    return arr
end

function _t2c_empty(value)
    value isa AbstractString && return isempty(value)
    try
        return length(value) == 0
    catch
        return false
    end
end

function _t2c_format_value(value, key="")
    value isa AbstractString && return "\"" * String(value) * "\""
    value isa Bool && return string(Int(value))
    value isa Integer && return string(value)
    value isa AbstractFloat && return _t2c_format_float(value)
    if key == "physicsparamcases" && value isa AbstractMatrix
        rows = String[]
        for i in 1:size(value, 1)
            push!(rows, join([_t2c_format_float(Float64(v)) for v in value[i, :]], " "))
        end
        return "[" * join(rows, "; ") * "]"
    end
    vals = collect(vec(value))
    isempty(vals) && return "[]"
    if all(v -> v isa AbstractString, vals)
        return "[" * join(["\"" * String(v) * "\"" for v in vals], ", ") * "]"
    end
    return "[" * join([_t2c_format_float(Float64(v)) for v in vals], ", ") * "]"
end

_t2c_format_float(value) = string(Float64(value))

function _t2c_write_readme(dest)
    text = """
# Exasim Text2Code Export

This directory contains high-level Text2Code inputs exported from an Exasim frontend.

Generated files:

- `pdemodel.txt`: PDE model definition consumed by Text2Code.
- `pdeapp.txt`: application, mesh, solver, output, and runtime configuration.
- `grid.bin`: mesh coordinates and connectivity.
- `xdg.bin`, `udg.bin`, `vdg.bin`, `wdg.bin`: optional field data written only when present.

Regenerate the application with:

```sh
/path/to/exasim-prefix/bin/text2code pdeapp.txt
```

The `vdg.bin` file stores external variables. In backend data structures these are also called `odg`.
"""
    open(joinpath(dest, "README.md"), "w") do io
        write(io, text)
    end
end
