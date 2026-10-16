# [nonmatching] Phase 9: interfacetype field, validation and app.bin slot 34 (Julia frontend).
#
# Run: julia --startup-file=no tests/frontends/julia/test_interfacetype.jl   (exit 0 pass, 1 fail)
#
# The preprocessing files are included into a scratch module so the test needs no
# package dependencies. The parameter-sweep template is extracted from
# Postprocessing/exasim.jl by parsing the file, so the production code itself is
# exercised. The text2code exporter is checked by its key list and the exported
# line format is round-tripped through text2code by tests/text2code/interfacetype-test.sh.

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const SRC = joinpath(ROOT, "frontends", "Julia", "Exasim", "src")

module P9Frontend
include(joinpath(Main.SRC, "Preprocessing", "Types.jl"))
using .Types
include(joinpath(Main.SRC, "Preprocessing", "initializepde.jl"))
include(joinpath(Main.SRC, "Preprocessing", "writeapp.jl"))
end

failures = String[]
check(condition, message) = condition || (push!(failures, message); println("FAIL: ", message); false)

function base_pde()
    pde = P9Frontend.initializepde("Backend")
    for f in (:nd, :ncx, :nc, :ncu, :ncq, :ncw, :nco, :nch, :ncp, :nce, :mpiprocs)
        setfield!(pde, f, 1)
    end
    return pde
end

# app.bin layout: numsize, nsize[numsize], ndims[nsize1], flag[nsize2], problem[nsize3].
function read_flag_problem(filename)
    values = reinterpret(Float64, read(filename))
    numsize = Int(values[1])
    nsize = Int.(values[2:1+numsize])
    start = 2 + numsize + nsize[1]
    flag = collect(values[start:start+nsize[2]-1])
    start += nsize[2]
    problem = collect(values[start:start+nsize[3]-1])
    return flag, problem
end

function write_app(pde, dir, name)
    filename = joinpath(dir, name)
    app = P9Frontend.writeapp(deepcopy(pde), filename)
    return app, read_flag_problem(filename)
end

mktempdir() do tmp
    pde = base_pde()
    check(pde.interfacetype == 0, "initializepde default interfacetype is 0")
    _, (_, problem) = write_app(pde, tmp, "default.bin")
    check(length(problem) >= 35 && problem[35] == 0.0, "default interfacetype writes 0 at zero-based slot 34")

    for value in (0, 1)
        pde = base_pde()
        pde.interfacetype = value
        _, (_, problem) = write_app(pde, tmp, "explicit.bin")
        check(problem[35] == Float64(value), "interfacetype=$value written at zero-based slot 34")
    end

    # The struct field is Int64, so non-integers cannot be assigned; out-of-range integers are rejected.
    for value in (2, -1)
        pde = base_pde()
        pde.interfacetype = value
        filename = joinpath(tmp, "invalid.bin")
        isfile(filename) && rm(filename)
        rejected = try
            P9Frontend.writeapp(deepcopy(pde), filename)
            false
        catch err
            occursin("interfacetype", sprint(showerror, err))
        end
        check(rejected, "interfacetype=$value is rejected with a message naming the field")
        check(!isfile(filename), "interfacetype=$value writes no app.bin")
    end
    pde = base_pde()
    check(try (pde.interfacetype = 0.5; false) catch; true end, "non-integer interfacetype cannot be assigned")

    # User extras begin at zero-based slot 35 and keep their order.
    pde = base_pde()
    pde.interfacetype = 1
    pde.problem = [7 8 9]
    _, (_, problem) = write_app(pde, tmp, "extras.bin")
    check(length(problem) == 38, "35 fixed slots plus three user extras")
    check(problem[35] == 1.0 && problem[36:end] == [7.0, 8.0, 9.0], "user extras follow slot 34 in order")

    # Parameter-sweep template round trip (Postprocessing/exasim.jl writeapptemplate).
    exprs = Meta.parseall(read(joinpath(SRC, "Postprocessing", "exasim.jl"), String)).args
    defs = filter(e -> e isa Expr && e.head == :function &&
                       e.args[1] isa Expr && e.args[1].args[1] == :writeapptemplate, exprs)
    check(length(defs) == 1, "writeapptemplate found in Postprocessing/exasim.jl")
    template_fn = Core.eval(P9Frontend, defs[1])
    pde = base_pde()
    pde.uniformrefinementlevel = 1
    pde.problem = [7 8]
    pde.flag = reshape([5], 1, 1)
    first, (flag1, problem1) = write_app(pde, tmp, "sweep1.bin")
    template = Base.invokelatest(template_fn, first)
    second, (flag2, problem2) = write_app(template, tmp, "sweep2.bin")
    check(problem2 == problem1, "sweep template round trip preserves the problem vector " *
          "($(length(problem1)) -> $(length(problem2)) values)")
    check(flag2 == flag1, "sweep template round trip preserves the flag vector ($(length(flag1)) -> $(length(flag2)))")
    check(vec(second.factor) == vec(first.factor), "sweep template round trip preserves factor")
    check(vec(second.solversparam) == vec(first.solversparam), "sweep template round trip preserves solversparam")

    exporter = read(joinpath(SRC, "Gencode", "exporttext2code.jl"), String)
    check(occursin("\"interfacefluxmap\", \"interfacetype\"", exporter), "text2code exporter lists interfacetype")
end

if isempty(failures)
    println("PASS: test_interfacetype.jl")
else
    println("FAIL: test_interfacetype.jl ($(length(failures)) failures)")
    exit(1)
end
