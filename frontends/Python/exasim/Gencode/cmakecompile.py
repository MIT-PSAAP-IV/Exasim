import os
import shutil
import subprocess

from .. import config
from ..exasim_path import exasim_path


def cmakecompile(pde):
    print("Compile C++ Exasim code using cmake...")

    cdir = os.getcwd()

    if pde['sharedbuild'] == 0:
        sourcepath = os.path.join(exasim_path(), "examples", "exasimfe")
        targetpath = os.path.join(cdir, "exasim")
        os.makedirs(targetpath, exist_ok=True)

        for name in ("exasimfeapp.cpp", "CMakeLists.txt", "frontendprovider.cpp"):
            shutil.copy2(os.path.join(sourcepath, name), os.path.join(targetpath, name))

    sourcepath = pde['builddir']
    buildpath = os.path.join(pde['builddir'], "build")
    os.makedirs(buildpath, exist_ok=True)

    exe = os.path.join(buildpath, "exasimapp")
    if os.path.exists(exe):
        os.remove(exe)

    cfg = [
        "cmake",
        "-S", sourcepath,
        "-B", ".",
        "-D", "EXASIM_MPI=" + ("OFF" if pde['mpiprocs'] == 1 else "ON"),
    ]

    if pde['platform'] == "gpu":
        cfg.extend(["-D", "EXASIM_CUDA=ON"])
    elif pde['platform'] == "hip":
        cfg.extend(["-D", "CMAKE_CXX_COMPILER=hipcc"])
        cfg.extend(["-D", "EXASIM_HIP=ON"])

    pde['exasimpath'] = str(config.install_prefix())
    cfg.extend(["-D", "Exasim_DIR=" + pde['exasimpath']])

    subprocess.run(cfg, cwd=buildpath, check=True)
    subprocess.run(
        ["cmake", "--build", ".", "--target", "exasimapp", "--verbose"],
        cwd=buildpath,
        check=True,
    )
    return cfg
