const IntP = Int64
const FloatP = Float64
const root = normpath(joinpath(@__DIR__, "..", "..", "frontends", "Julia", "Exasim", "src"))
include(joinpath(root, "Preprocessing", "initializepde.jl"))
include(joinpath(root, "Preprocessing", "checkconformalinterface.jl"))
include(joinpath(root, "Preprocessing", "writeapp.jl"))
include(joinpath(root, "Gencode", "exporttext2code.jl"))
base = initializepde("")
@assert base.compositehdg == 0 && base.compositebatchsize == 1024
mktempdir() do directory
    for cap in (1, 127, 128, 256, 512, 1024, 2048, 4096)
        app = deepcopy(base)
        app.compositehdg = 1
        app.compositebatchsize = cap
        app.interfacetype = 0
        app.uniformrefinementlevel = 2
        filename = joinpath(directory, "app.bin")
        writeapp(app, filename)
        data = reinterpret(Float64, read(filename))
        sizes = Int.(data[2:1+Int(data[1])])
        start = length(sizes) + 2
        @assert data[start+23] == 1 && data[start+24] == cap && data[start+22] == 0
        problem = start + sizes[1] + sizes[2]
        @assert data[problem+33] == 2 && data[problem+34] == 0
        text = joinpath(directory, "pdeapp.txt")
        mesh = Dict(:boundarycondition => Int[], :boundaryexpr => String[])
        _t2c_write_pdeapp(app, mesh, Dict("meshfile" => "mesh.bin"), text)
        @assert occursin("compositehdg = 1;", read(text, String))
        @assert occursin("compositebatchsize = $(cap);", read(text, String))
    end
    for codes in (5, [1001, 5, 5], Int[])
        app = deepcopy(base); app.compositebc = codes
        filename = joinpath(directory, "boundary.bin")
        writeapp(app,filename)
        data = reinterpret(Float64,read(filename))
        start = 2 + Int(data[1])
        expected = codes isa Real ? [codes] : sort(unique(codes))
        @assert data[start+25] == length(expected)
        @assert data[start+40:start+39+length(expected)] == expected
        text = joinpath(directory,"boundary.txt")
        _t2c_write_pdeapp(app,Dict(:boundarycondition => Int[], :boundaryexpr => String[]),Dict("meshfile" => "mesh.bin"),text)
        isempty(expected) || @assert occursin("compositebc =",read(text,String))
    end
    unsupported = deepcopy(base)
    unsupported.compositehdg = 1
    unsupported.interfacetype = 1
    rejected = false
    try
        writeapp(unsupported, joinpath(directory, "nonmatching.bin"))
    catch error
        rejected = occursin("not available in this source release", sprint(showerror, error))
    end
    @assert rejected
    for (name, values) in ((:compositebc, (0, -1, 5.5, NaN, Inf, 2147483648, [5, NaN], true)),
                           (:compositehdg, (-1, 2, 0.5, NaN, Inf)),
                           (:compositebatchsize, (0, -1, 1024.5, NaN, Inf, 2147483648)))
        for value in values
            app = deepcopy(base)
            rejected = false
            try
                setproperty!(app, name, value)
                writeapp(app, joinpath(directory, "invalid.bin"))
            catch
                rejected = true
            end
            @assert rejected
        end
    end
end
println("COMPOSITE_JULIA_FRONTEND_PASS")
