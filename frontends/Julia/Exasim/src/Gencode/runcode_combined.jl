# Run the combined multi-PDE executable built by cmakecompile_combined.
# Emits the per-model CLI the solver's ParseInputs expects:
#   exasimapp N datain<s0>/ dataout<s0>/out datain<s1>/ dataout<s1>/out ...
# one (datain, dataout) pair per model slot (nested strn dirs; "" for slot 0).
# Mirrors the Python frontend's runcode_combined.
function runcode_combined(pdes, domainranks=nothing)
    display("Run combined multi-PDE C++ Exasim code ...")

    p0 = pdes[1]
    n = length(pdes)
    exe = joinpath(p0.builddir, "combined", "build", "exasimapp")
    if !isfile(exe)
        error("Combined executable not found at $exe; run Gencode.cmakecompile_combined(pdes) first.")
    end

    datapath = p0.datapath
    args = String[string(n)]
    for p in pdes
        strn = model_strn(p)
        push!(args, joinpath(datapath, "datain", strn) * "/")
        push!(args, joinpath(datapath, "dataout", strn, "out"))
    end

    totalranks = p0.mpiprocs
    if !isnothing(domainranks)
        counts = Int.(collect(domainranks))
        length(counts) == n && all(counts .> 0) ||
            error("domainranks must contain one positive integer count per domain")
        totalranks = sum(counts)
        push!(args, "--domain-ranks")
        push!(args, join(counts, ","))
    end

    if totalranks == 1
        cmd = `$exe $args`
    else
        mpirun = p0.mpirun
        mpitxt = joinpath(p0.builddir, "combined", "build", "mpiexec.txt")
        if isfile(mpitxt)
            discovered = strip(read(mpitxt, String))
            isempty(discovered) || (mpirun = discovered)
        end
        cmd = `$mpirun -np $totalranks $exe $args`
    end

    run(Cmd(cmd; dir=datapath))
    return string(cmd)
end

function runcode_coupled(pdes)
    length(pdes) == 2 || error("Coupled interface solves support exactly two models.")
    p0 = pdes[1]
    exe = joinpath(p0.builddir, "combined", "build", "exasimapp")
    isfile(exe) || error("Combined executable not found at $exe; run Gencode.cmakecompile_combined(pdes) first.")
    args = String[]
    for p in pdes
        strn = model_strn(p)
        push!(args, joinpath(p0.datapath, "datain", strn) * "/")
        push!(args, joinpath(p0.datapath, "dataout", strn, "out"))
    end
    mpirun = p0.mpirun
    mpitxt = joinpath(p0.builddir, "combined", "build", "mpiexec.txt")
    if isfile(mpitxt)
        discovered = strip(read(mpitxt, String)); isempty(discovered) || (mpirun = discovered)
    end
    nprocs = pdes[1].mpiprocs + pdes[2].mpiprocs
    cmd = `$mpirun -np $nprocs $exe $(100 + pdes[1].mpiprocs) $args`
    run(Cmd(cmd; dir=p0.datapath))
    return string(cmd)
end
