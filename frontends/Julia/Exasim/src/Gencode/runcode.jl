# Run the solver executable built by cmakecompile.
#
# Uses the legacy CLI: exasimapp <numpde> <datain>/ <dataout>/out, with
# datain/dataout under pde.datapath and the executable in the hidden
# pde.builddir. Julia's `run` throws on a nonzero solver exit code.
function runcode(pde, numpde=1)

display("Run C++ Exasim code ...")

exe = joinpath(model_builddir(pde), "build", "exasimapp")
if !isfile(exe)
    error("Solver executable not found at $exe; run Gencode.cmakecompile(pde) first.")
end

# per-model datain/dataout (nested strn dirs; "" for model 0 -> datain/)
strn = model_strn(pde)
datain = joinpath(pde.datapath, "datain", strn) * "/"
dataout = joinpath(pde.datapath, "dataout", strn, "out")

if pde.mpiprocs == 1
    cmd = `$exe $numpde $datain $dataout`
else
    # Prefer the MPI launcher CMake discovered at build-configure time
    # (portable); fall back to the frontend-detected pde.mpirun.
    mpirun = pde.mpirun
    mpitxt = joinpath(pde.builddir, "build", "mpiexec.txt")
    if isfile(mpitxt)
        discovered = strip(read(mpitxt, String))
        isempty(discovered) || (mpirun = discovered)
    end
    cmd = `$mpirun -np $(pde.mpiprocs) $exe $numpde $datain $dataout`
end

run(Cmd(cmd; dir=pde.datapath))

return string(cmd)
end
