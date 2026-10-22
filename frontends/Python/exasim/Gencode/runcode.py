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

    executionmode = pde.get('executionmode', 0)
    if executionmode == 0:
        modearg = None
    elif executionmode == 1:
        modearg = "postprocess"
    else:
        raise ValueError("Unsupported executionmode={}. Use 0 for solve or "
                         "1 for postprocess.".format(executionmode))

    # datain/dataout are isolated per model by strn (preprocessing writes the
    # same way); strn="" for model 0 collapses to the historical paths.
    strn = config.model_strn(pde)
    datain = os.path.join(pde['datapath'], "datain", strn, "")
    dataoutdir = pde.get('dataoutpath') or os.path.join(pde['datapath'], "dataout", strn)
    dataout = os.path.join(dataoutdir, "out")

    if pde['mpiprocs'] == 1:
        cmd = [exe, str(numpde), datain, dataout]
        if modearg is not None:
            cmd.insert(1, modearg)
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
        if modearg is not None:
            cmd.insert(4, modearg)

    start_time = time.time()
    subprocess.run(cmd, cwd=pde['datapath'], check=True)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")

    return " ".join(cmd)


def runcode_combined(pdes, domain_ranks=None):
    """Run the combined multi-PDE executable built by cmakecompile_combined.

    Emits the per-model CLI the solver's ParseInputs expects:
        exasimapp N datain0/ dataout0/out datain1/ dataout1/out ...
    one (datain, dataout) pair per model slot, each in its model's strn subdir
    (matching preprocessing). Raises on a nonzero solver exit code.
    """
    print("Run combined multi-PDE C++ Exasim code ...")

    p0 = pdes[0]
    n = len(pdes)
    exe = os.path.join(p0['builddir'], "combined", "build", "exasimapp")
    if not os.path.exists(exe):
        raise RuntimeError(f"Combined executable not found at {exe}; "
                           "run Gencode.cmakecompile_combined(pdes) first.")

    datapath = p0['datapath']
    pairs = []
    for p in pdes:
        strn = config.model_strn(p)
        pairs.append(os.path.join(datapath, "datain", strn, ""))
        pairs.append(os.path.join(datapath, "dataout", strn, "out"))

    total_ranks = p0['mpiprocs']
    if domain_ranks is not None:
        if (len(domain_ranks) != n or
            any(isinstance(value, bool) or int(value) != value or int(value) <= 0
                for value in domain_ranks)):
            raise ValueError("domain_ranks must contain one positive integer count per domain")
        domain_ranks = [int(value) for value in domain_ranks]
        total_ranks = sum(domain_ranks)
        pairs += ["--domain-ranks", ",".join(str(value) for value in domain_ranks)]

    if total_ranks == 1:
        cmd = [exe, str(n)] + pairs
    else:
        mpirun = p0['mpirun']
        mpitxt = os.path.join(p0['builddir'], "combined", "build", "mpiexec.txt")
        if os.path.exists(mpitxt):
            with open(mpitxt) as f:
                discovered = f.read().strip()
            if discovered:
                mpirun = discovered
        cmd = [mpirun, "-np", str(total_ranks), exe, str(n)] + pairs

    start_time = time.time()
    subprocess.run(cmd, cwd=datapath, check=True)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")

    return " ".join(cmd)


def runcode_coupled(pdes):
    """Run a two-domain interface-coupled solve with the combined executable.

    The solver's split-domain CLI is
        mpirun -np <np1+np2> exasimapp <100+np1> datain1/ dataout1/out datain2/ dataout2/out
    ranks [0, np1) solve model 1 and the rest model 2. A coupled world is an
    MPI application, so mpirun is used even with one rank per model.
    """
    print("Run coupled two-domain C++ Exasim code ...")

    if len(pdes) != 2:
        raise ValueError("Coupled interface solves support exactly two models.")
    p0 = pdes[0]
    exe = os.path.join(p0['builddir'], "combined", "build", "exasimapp")
    if not os.path.exists(exe):
        raise RuntimeError(f"Combined executable not found at {exe}; "
                           "run Gencode.cmakecompile_combined(pdes) first.")

    datapath = p0['datapath']
    pairs = []
    for p in pdes:
        strn = config.model_strn(p)
        pairs.append(os.path.join(datapath, "datain", strn, ""))
        pairs.append(os.path.join(datapath, "dataout", strn, "out"))

    mpirun = p0['mpirun']
    mpitxt = os.path.join(p0['builddir'], "combined", "build", "mpiexec.txt")
    if os.path.exists(mpitxt):
        with open(mpitxt) as f:
            discovered = f.read().strip()
        if discovered:
            mpirun = discovered
    nprocs = pdes[0]['mpiprocs'] + pdes[1]['mpiprocs']
    cmd = [mpirun, "-np", str(nprocs), exe, str(100 + pdes[0]['mpiprocs'])] + pairs

    start_time = time.time()
    subprocess.run(cmd, cwd=datapath, check=True)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")

    return " ".join(cmd)
