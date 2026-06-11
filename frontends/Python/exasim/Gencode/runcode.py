import os
import subprocess
import time


def runcode(pde, numpde):
    """Run the solver executable built by cmakecompile.

    Uses the legacy CLI: exasimapp <numpde> <datain>/ <dataout>/out, with
    datain/dataout under pde['datapath'] and the executable in the hidden
    pde['builddir']. Raises on a nonzero solver exit code.
    """
    print("Run C++ Exasim code ...")

    exe = os.path.join(pde['builddir'], "build", "exasimapp")
    if not os.path.exists(exe):
        raise RuntimeError(f"Solver executable not found at {exe}; "
                           "run Gencode.cmakecompile(pde) first.")

    datain = os.path.join(pde['datapath'], "datain") + "/"
    dataout = os.path.join(pde['datapath'], "dataout", "out")

    if pde['mpiprocs'] == 1:
        cmd = [exe, str(numpde), datain, dataout]
    else:
        cmd = [pde['mpirun'], "-np", str(pde['mpiprocs']),
               exe, str(numpde), datain, dataout]

    start_time = time.time()
    subprocess.run(cmd, cwd=pde['datapath'], check=True)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")

    return " ".join(cmd)
