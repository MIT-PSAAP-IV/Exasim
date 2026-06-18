# Run the combined multi-PDE executable built by cmakecompile_combined.
# Emits the per-model CLI the solver's ParseInputs expects:
#   exasimapp N datain<s0>/ dataout<s0>/out datain<s1>/ dataout<s1>/out ...
# one (datain, dataout) pair per model slot (nested strn dirs; "" for slot 0).
# Mirrors the Python frontend's runcode_combined.
function runcode_combined(pdes)
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

    if p0.mpiprocs == 1
        cmd = `$exe $args`
    else
        mpirun = p0.mpirun
        mpitxt = joinpath(p0.builddir, "combined", "build", "mpiexec.txt")
        if isfile(mpitxt)
            discovered = strip(read(mpitxt, String))
            isempty(discovered) || (mpirun = discovered)
        end
        cmd = `$mpirun -np $(p0.mpiprocs) $exe $args`
    end

    run(Cmd(cmd; dir=datapath))
    return string(cmd)
end
