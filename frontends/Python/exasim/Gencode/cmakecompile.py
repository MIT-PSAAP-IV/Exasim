import os
import subprocess

from .. import config


def _render(template_path, dest_path, subs):
    text = open(template_path).read()
    for key, val in subs.items():
        text = text.replace("@" + key + "@", str(val))
    # Write only on change so unchanged apps don't dirty mtimes (recompiles).
    if os.path.exists(dest_path):
        with open(dest_path) as f:
            if f.read() == text:
                return
    with open(dest_path, "w") as f:
        f.write(text)


def cmakecompile(pde):
    """Build the solver executable for the generated model.

    Renders the installed frontend-app templates into the hidden
    pde['builddir'] and builds them against the installed Exasim package via
    find_package(Exasim). The gencode step must have written the kernel set
    to <builddir>/kernels first. Returns the configure command.
    """
    print("Compile C++ Exasim code against the installed Exasim package...")

    prefix = config.install_prefix()
    pde['exasimpath'] = str(prefix)

    builddir = pde['builddir']
    kernels = os.path.join(builddir, "kernels")
    if not os.path.isdir(kernels):
        raise RuntimeError(
            f"No generated kernels at {kernels}; run Gencode.gencode(pde) first.")

    if pde['platform'] == "gpu":
        variant = "gpumpi" if pde['mpiprocs'] > 1 else "gpu"
    else:
        variant = "cpumpi" if pde['mpiprocs'] > 1 else "cpu"

    tmpl = config.frontend_app_template_dir()
    subs = {
        "EXASIM_VARIANT": variant,
        "MODEL_ID": pde['modelid'],
        "KERNEL_DIR": kernels,
    }
    _render(tmpl / "CMakeLists.txt.in", os.path.join(builddir, "CMakeLists.txt"), subs)
    _render(tmpl / "main.cpp.in", os.path.join(builddir, "main.cpp"), subs)

    bdir = os.path.join(builddir, "build")
    exe = os.path.join(bdir, "exasimapp")
    cmake = config.cmake_command()
    cfg = [cmake, "-S", builddir, "-B", bdir,
           "-DExasim_DIR=" + str(config.cmake_dir())]

    subprocess.run(cfg, check=True)
    jobs = os.environ.get("JOBS") or str(os.cpu_count() or 4)
    subprocess.run([cmake, "--build", bdir, "--parallel", jobs], check=True)

    if not os.path.exists(exe):
        raise RuntimeError(f"Build did not produce {exe}.")
    return cfg
