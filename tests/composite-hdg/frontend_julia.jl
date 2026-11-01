const IntP = Int64
const FloatP = Float64
const root = normpath(joinpath(@__DIR__, "..", "..", "frontends", "Julia", "Exasim", "src"))
include(joinpath(root, "Preprocessing", "initializepde.jl"))
include(joinpath(root, "Preprocessing", "writeapp.jl"))
include(joinpath(root, "Gencode", "exporttext2code.jl"))
base = initializepde("")
@assert base.compositehdg == 0 && base.compositebatchsize == 1024
mktempdir() do directory
    for cap in (1, 127, 128, 256, 512, 1024, 2048, 4096)
        app = deepcopy(base)
        app.compositehdg = 1
        app.compositebatchsize = cap
        app.interfacetype = 1
        app.uniformrefinementlevel = 2
        filename = joinpath(directory, "app.bin")
        writeapp(app, filename)
        data = reinterpret(Float64, read(filename))
        sizes = Int.(data[2:1+Int(data[1])])
        start = length(sizes) + 2
        @assert data[start+23] == 1 && data[start+24] == cap && data[start+22] == 0
        problem = start + sizes[1] + sizes[2]
        @assert data[problem+33] == 2 && data[problem+34] == 1
        text = joinpath(directory, "pdeapp.txt")
        mesh = Dict(:boundarycondition => Int[], :boundaryexpr => String[])
        _t2c_write_pdeapp(app, mesh, Dict("meshfile" => "mesh.bin"), text)
        @assert occursin("compositehdg = 1;", read(text, String))
        @assert occursin("compositebatchsize = $(cap);", read(text, String))
    end
    for (name, values) in ((:compositehdg, (-1, 2, 0.5, NaN, Inf)),
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
