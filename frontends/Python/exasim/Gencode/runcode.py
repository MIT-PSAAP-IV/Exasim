import os
import subprocess
import time

from .. import config


def runcode(pde, numpde):
    """Run the solver executable built by cmakecompile.

    Uses the legacy CLI: exasimapp <numpde> <datain>/ <dataout>/out, with
    datain/dataout under pde['datapath'] and the executable in the per-model
    build dir. Raises on a nonzero solver exit code.
    """
    print("Run C++ Exasim code ...")

    exe = os.path.join(config.model_builddir(pde), "build", "exasimapp")
    if not os.path.exists(exe):
        raise RuntimeError(f"Solver executable not found at {exe}; "
                           "run Gencode.cmakecompile(pde) first.")

    # datain/dataout are isolated per model by strn (preprocessing writes the
    # same way); strn="" for model 0 collapses to the historical paths.
    strn = config.model_strn(pde)
    datain = os.path.join(pde['datapath'], "datain", strn, "")
    dataout = os.path.join(pde['datapath'], "dataout", strn, "out")

    if pde['mpiprocs'] == 1:
        cmd = [exe, str(numpde), datain, dataout]
    else:
        # Prefer the MPI launcher CMake discovered at build-configure time
        # (portable); fall back to the frontend-detected pde['mpirun'].
        mpirun = pde['mpirun']
        mpitxt = os.path.join(pde['builddir'], "build", "mpiexec.txt")
        if os.path.exists(mpitxt):
            with open(mpitxt) as f:
                discovered = f.read().strip()
            if discovered:
                mpirun = discovered
        cmd = [mpirun, "-np", str(pde['mpiprocs']),
               exe, str(numpde), datain, dataout]

    start_time = time.time()
    subprocess.run(cmd, cwd=pde['datapath'], check=True)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")

    return " ".join(cmd)
