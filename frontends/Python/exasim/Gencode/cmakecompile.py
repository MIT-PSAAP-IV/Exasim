import glob
import hashlib
import os
import shutil
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

    modelid = config.resolve_modelid(pde)
    tmpl = config.frontend_app_template_dir()
    subs = {
        "EXASIM_VARIANT": variant,
        "MODEL_ID": modelid,
        "KERNEL_DIR": kernels,
    }
    _render(tmpl / "CMakeLists.txt.in", os.path.join(builddir, "CMakeLists.txt"), subs)
    _render(tmpl / "main.cpp.in", os.path.join(builddir, "main.cpp"), subs)

    bdir = os.path.join(builddir, "build")
    exe = os.path.join(bdir, "exasimapp")
    cmake = config.cmake_command()
    cfg = [cmake, "-S", builddir, "-B", bdir,
           "-DExasim_DIR=" + str(config.cmake_dir())]

    # Hash the model inputs (kernels + app templates + substitution values +
    # the install). Templates rather than rendered files: the rendered sources
    # embed absolute paths, which would make the digest directory-specific and
    # defeat the cross-directory model cache.
    # stamp = os.path.join(bdir, ".exasim_model_hash")
    # digest = _model_hash(kernels,
    #                      [tmpl / "CMakeLists.txt.in", tmpl / "main.cpp.in"],
    #                      prefix, variant, pde['modelid'])
    # if os.path.exists(exe) and os.path.exists(stamp):
    #     with open(stamp) as f:
    #         if f.read() == digest:
    #             print("Model unchanged (hash match); skipping build.")
    #             return cfg

    # Per-user model cache: the relocatable (libfrontend_model, exasimapp)
    # pair built for this modelid+digest by any earlier app run.
    # cachedir = os.path.join(_cache_root(), str(pde['modelid']), digest)
    # cached = _cache_files(cachedir)
    # if cached:
    #     os.makedirs(bdir, exist_ok=True)
    #     for src in cached:
    #         shutil.copy2(src, os.path.join(bdir, os.path.basename(src)))
    #     with open(stamp, "w") as f:
    #         f.write(digest)
    #     print(f"Model cache hit ({cachedir}); skipping build.")
    #     return cfg

    subprocess.run(cfg, check=True)
    jobs = os.environ.get("JOBS") or str(os.cpu_count() or 4)
    subprocess.run([cmake, "--build", bdir, "--parallel", jobs], check=True)

    if not os.path.exists(exe):
        raise RuntimeError(f"Build did not produce {exe}.")
    # with open(stamp, "w") as f:
    #     f.write(digest)

    # Populate the cache for other app directories / future runs.
    # libs = glob.glob(os.path.join(bdir, "libfrontend_model.*"))
    # if libs:
    #     os.makedirs(cachedir, exist_ok=True)
    #     for src in libs + [exe]:
    #         shutil.copy2(src, os.path.join(cachedir, os.path.basename(src)))
    return cfg


def _cache_root():
    """Per-user cache of built model libraries (EXASIM_CACHE_DIR overrides)."""
    root = os.environ.get("EXASIM_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".exasim")
    return os.path.join(root, "cache")


def _cache_files(cachedir):
    """Return the cached (lib, exe) pair, or None if incomplete."""
    exe = os.path.join(cachedir, "exasimapp")
    libs = glob.glob(os.path.join(cachedir, "libfrontend_model.*"))
    if libs and os.path.exists(exe):
        return libs + [exe]
    return None


def _model_hash(kernelsdir, extra_files, prefix, variant, modelid):
    """SHA-256 over the kernel set, the app templates, the variant/model ID,
    and the identity of the Exasim install they build against."""
    h = hashlib.sha256()
    h.update(f"{variant}|{modelid}".encode())
    for name in sorted(os.listdir(kernelsdir)):
        h.update(name.encode())
        with open(os.path.join(kernelsdir, name), "rb") as f:
            h.update(f.read())
    for path in extra_files:
        with open(path, "rb") as f:
            h.update(f.read())
    h.update(str(prefix).encode())
    targets = os.path.join(prefix, "lib", "cmake", "Exasim", "ExasimTargets.cmake")
    if os.path.exists(targets):
        st = os.stat(targets)
        h.update(f"{st.st_mtime_ns}:{st.st_size}".encode())
    return h.hexdigest()
