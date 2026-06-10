# Run the solver executable built by cmakecompile.
#
# Uses the legacy CLI: exasimapp <numpde> <datain>/ <dataout>/out, with
# datain/dataout under pde.datapath and the executable in the hidden
# pde.builddir. Julia's `run` throws on a nonzero solver exit code.
function runcode(pde, numpde=1)

display("Run C++ Exasim code ...")

exe = joinpath(pde.builddir, "build", "exasimapp")
if !isfile(exe)
    error("Solver executable not found at $exe; run Gencode.cmakecompile(pde) first.")
end

datain = joinpath(pde.datapath, "datain") * "/"
dataout = joinpath(pde.datapath, "dataout", "out")

if pde.mpiprocs == 1
    cmd = `$exe $numpde $datain $dataout`
else
    mpirun = pde.mpirun
    cmd = `$mpirun -np $(pde.mpiprocs) $exe $numpde $datain $dataout`
end

run(Cmd(cmd; dir=pde.datapath))

return string(cmd)
end
