import json
import os
import shutil
import subprocess
import tempfile
import numpy

from .. import config
from .cmakecompile import _render


def _variant(pde):
    """The Exasim solver variant implied by platform + process count."""
    if pde['platform'] == "gpu":
        return "gpumpi" if pde['mpiprocs'] > 1 else "gpu"
    return "cpumpi" if pde['mpiprocs'] > 1 else "cpu"


def _copytree(src, dst):
    """Mirror src into dst (replacing dst), preserving file metadata."""
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def exportapp(pde, dest=None, build=True):
    """Package a relocatable "data transfer app" bundle.

    Collects everything the solver needs to run on any machine that has an
    Exasim install: the binary inputs (datain/, plus an empty dataout/), the
    generated kernel .cpp set, a relocatable CMakeLists.txt, main.cpp, a
    run.sh, a manifest, and a copy of the source model. The bundle depends on
    an Exasim install at run time (find_package(Exasim)) but carries no
    absolute paths, so it can be copied anywhere.

    Boundary conditions need no text form here: preprocessing has already
    resolved them into the binary datain. Requires preprocessing (datain) and
    gencode (kernels) to have run already.

    With build=True (default) the bundle is configured, built, and run in
    place against the local Exasim install to verify it works before hand-off.

    Returns the absolute bundle path.
    """
    if dest is None:
        dest = pde.get('exportapp')
    if not dest:
        raise RuntimeError(
            "exportapp: no destination (set pde['exportapp'] or pass dest).")
    dest = os.path.abspath(dest)

    builddir = pde['builddir']
    kernels = os.path.join(builddir, "kernels")
    if not os.path.isdir(kernels):
        raise RuntimeError(
            f"No generated kernels at {kernels}; run Gencode.gencode(pde) first.")

    datain = os.path.join(pde['datapath'], "datain")
    if not os.path.isdir(datain):
        raise RuntimeError(
            f"No datain at {datain}; run Preprocessing.preprocessing(pde, mesh) first.")

    print(f"Export Exasim data-transfer app to {dest} ...")
    os.makedirs(dest, exist_ok=True)

    # 1. Runtime inputs + the output directory the solver writes into.
    _copytree(datain, os.path.join(dest, "datain"))
    os.makedirs(os.path.join(dest, "dataout"), exist_ok=True)
    _write_physicsparamcases_if_needed(pde, os.path.join(dest, "datain"))

    # 2. The generated kernel .cpp set.
    _copytree(kernels, os.path.join(dest, "kernels"))

    # 3 + 4. Relocatable CMakeLists.txt + main.cpp from the shared templates.
    # KERNEL_DIR resolves inside the bundle at configure time (not an absolute
    # path), which is what makes the bundle relocatable. The Exasim install is
    # the only external dependency, supplied on the target via Exasim_DIR.
    variant = _variant(pde)
    numpde = 1
    modelid = config.resolve_modelid(pde)
    tmpl = config.frontend_app_template_dir()
    subs = {
        "EXASIM_VARIANT": variant,
        "MODEL_ID": str(modelid),
        "KERNEL_DIR": "${CMAKE_CURRENT_SOURCE_DIR}/kernels",
    }
    _render(str(tmpl / "CMakeLists.txt.in"),
            os.path.join(dest, "CMakeLists.txt"), subs)
    _render(str(tmpl / "main.cpp.in"),
            os.path.join(dest, "main.cpp"), subs)

    # 5. A copy of the source model, for reference / reproducibility.
    modelsrc = str(pde.get('modelfile', "")) + ".py"
    if os.path.isfile(modelsrc):
        shutil.copy2(modelsrc, os.path.join(dest, os.path.basename(modelsrc)))

    # 5b. A text2code DSL file (pdemodel.txt) emitted from the symbolic model,
    # so the bundle can regenerate kernels via text2code without the Python
    # frontend. Additive: if generation fails, warn and continue (the kernels
    # are already shipped in kernels/).
    try:
        from .genpdemodel import genpdemodel
        genpdemodel(pde, os.path.join(dest, "pdemodel.txt"))
    except Exception as exc:  # noqa: BLE001 - intentionally non-fatal
        print(f"WARNING: pdemodel.txt generation skipped ({exc!r}); "
              "the exported kernels/ are unaffected.")

    # 6. run.sh + manifest with the provenance needed to rebuild elsewhere.
    _write_runscript(dest, pde, variant, numpde)
    _write_manifest(dest, pde, variant, numpde)
    _write_readme(dest, variant, numpde)

    if build:
        _build_and_run(dest, pde, numpde)

    print(f"Exported data-transfer app: {dest}")
    return dest


def _physicsparam_sweep_cases(spec, nparam):
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)) and len(spec) == 0:
        return None
    if isinstance(spec, dict):
        if "samples" in spec:
            return _physicsparam_sweep_cases(spec["samples"], nparam)
        if "values" in spec:
            return _physicsparam_sweep_cases(spec["values"], nparam)
        if "grid" in spec:
            grid = spec["grid"]
            if len(grid) != nparam:
                raise ValueError("physicsparamsweep['grid'] must contain one value list per physics parameter.")
            meshes = numpy.meshgrid(*[numpy.asarray(v, dtype=float).ravel() for v in grid], indexing="ij")
            cases = numpy.column_stack([m.ravel() for m in meshes])
            return cases
        raise ValueError("physicsparamsweep dict must contain 'samples', 'values', or 'grid'.")

    arr = numpy.asarray(spec, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if nparam == 1:
            cases = arr.reshape((-1, 1))
        elif arr.size == nparam:
            cases = arr.reshape((1, nparam))
        else:
            raise ValueError(f"physicsparamsweep vector has {arr.size} entries; expected {nparam}.")
    elif arr.ndim == 2:
        cases = arr
    else:
        rows = []
        for i, row in enumerate(spec, start=1):
            values = numpy.asarray(row, dtype=float).ravel()
            if values.size != nparam:
                raise ValueError(f"physicsparamsweep case {i} has {values.size} parameters; expected {nparam}.")
            rows.append(values)
        cases = numpy.vstack(rows)

    if cases.shape[1] != nparam:
        raise ValueError(f"Each physicsparamsweep row must contain {nparam} physics parameters.")
    if not numpy.all(numpy.isfinite(cases)):
        raise ValueError("physicsparamsweep cases must contain finite numeric values.")
    return numpy.asarray(cases, dtype=numpy.float64)


def _write_physicsparamcases_if_needed(pde, datain):
    cases = _physicsparam_sweep_cases(pde.get("physicsparamsweep"), len(numpy.asarray(pde["physicsparam"]).ravel()))
    if cases is None or cases.shape[0] <= 1:
        return
    path = os.path.join(datain, "physicsparamcases.bin")
    header = numpy.asarray([cases.shape[0], cases.shape[1]], dtype=numpy.float64)
    with open(path, "wb") as f:
        header.tofile(f)
        numpy.asarray(cases, dtype=numpy.float64, order="C").ravel(order="C").tofile(f)


def _run_command(pde, numpde, exe, datain, dataout):
    """The argv the bundle uses to launch the solver (MPI-aware)."""
    if pde['mpiprocs'] > 1:
        return ["${MPIRUN:-mpirun}", "-np", str(pde['mpiprocs']),
                exe, str(numpde), datain, dataout]
    return [exe, str(numpde), datain, dataout]


def _write_runscript(dest, pde, variant, numpde):
    cmd = _run_command(pde, numpde, '"$here/build/exasimapp"',
                       '"$here/datain/"', '"$here/dataout/out"')
    runline = " ".join(cmd)
    text = f"""#!/usr/bin/env bash
# Build and run this exported Exasim data-transfer app.
# Requires an Exasim install on this machine; point EXASIM_ROOT at its prefix
# (the directory that contains lib/cmake/Exasim/). For MPI variants, set
# MPIRUN if your launcher is not "mpirun".
#
# The bundle was exported for variant "{variant}", but the kernels and datain
# are arch-independent: retarget to whatever the build machine has by setting
# EXASIM_VARIANT (cpu | cpumpi | gpu | gpumpi), e.g.
#   EXASIM_ROOT=/path/to/install EXASIM_VARIANT=gpu ./run.sh
set -euo pipefail
here="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
: "${{EXASIM_ROOT:?Set EXASIM_ROOT to your Exasim install prefix}}"
cmake -S "$here" -B "$here/build" \\
  -DExasim_DIR="$EXASIM_ROOT/lib/cmake/Exasim" \\
  -DEXASIM_VARIANT="${{EXASIM_VARIANT:-{variant}}}"
cmake --build "$here/build" --parallel
{runline}
"""
    path = os.path.join(dest, "run.sh")
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, 0o755)


def _write_manifest(dest, pde, variant, numpde):
    manifest = {
        "format": "exasim-data-transfer-app/1",
        "frontend": "python",
        "modelid": pde['modelid'],
        "variant": variant,
        "platform": pde['platform'],
        "mpiprocs": pde['mpiprocs'],
        "numpde": numpde,
        "hybrid": pde.get('hybrid', 0),
        "porder": pde.get('porder'),
        "built_against_prefix": str(config.install_prefix()),
    }
    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def _write_readme(dest, variant, numpde):
    text = f"""# Exasim data-transfer app

Self-contained bundle for the generated model (variant `{variant}`). It carries
the binary inputs, the generated kernels, and a relocatable CMake build, but
relies on an **Exasim install** on the machine where you build it.

## Contents
- `datain/`    binary solver inputs (mesh, master, app, solution); boundary
               conditions are already resolved here.
- `dataout/`   solver writes outputs here.
- `kernels/`   generated kernel .cpp set for this model.
- `CMakeLists.txt`, `main.cpp`   relocatable build of `exasimapp`.
- `run.sh`     build + run helper.
- `manifest.json`   provenance (model id, variant, process count, ...).

## Build & run
```sh
EXASIM_ROOT=/path/to/exasim/install ./run.sh
```
The bundle was exported for variant `{variant}`, but the kernels and datain are
arch-independent — retarget to whatever the build machine provides by setting
`EXASIM_VARIANT` (cpu | cpumpi | gpu | gpumpi):
```sh
EXASIM_ROOT=/path/to/install EXASIM_VARIANT=gpu ./run.sh
```
or manually:
```sh
cmake -S . -B build -DExasim_DIR=$EXASIM_ROOT/lib/cmake/Exasim -DEXASIM_VARIANT={variant}
cmake --build build --parallel
./build/exasimapp {numpde} datain/ dataout/out
```
"""
    with open(os.path.join(dest, "README.md"), "w") as f:
        f.write(text)


def exportapp_combined(pdes, dest=None, build=True):
    """Package a relocatable data-transfer bundle for a COMBINED multi-PDE app.

    Like exportapp() but for several generated models linked into one exasimapp.
    Each model m contributes its own datain<strn>/, dataout<strn>/ and
    kernels<strn>/ inside the bundle (strn="" for model 0), and the combined
    CMakeLists/main.cpp/run.sh carry the full id list and the N (datain,dataout)
    pairs. Single-model export is unaffected (use exportapp()).

    Returns the absolute bundle path.
    """
    if dest is None:
        dest = pdes[0].get('exportapp')
    if not dest:
        raise RuntimeError(
            "exportapp_combined: no destination (set pde[0]['exportapp'] or pass dest).")
    dest = os.path.abspath(dest)

    variant = _variant(pdes[0])
    ids = [config.resolve_modelid(p) for p in pdes]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"combined export needs distinct modelids; got {ids}")

    print(f"Export combined Exasim data-transfer app to {dest} ...")
    os.makedirs(dest, exist_ok=True)

    kdir_subs = []
    for p in pdes:
        strn = config.model_strn(p)
        kbuild = os.path.join(config.model_builddir(p), "kernels")
        if not os.path.isdir(kbuild):
            raise RuntimeError(
                f"No generated kernels at {kbuild}; run Gencode.gencode(pde) first.")
        src_datain = os.path.join(p['datapath'], "datain", strn) if strn \
            else os.path.join(p['datapath'], "datain")
        if not os.path.isdir(src_datain):
            raise RuntimeError(
                f"No datain at {src_datain}; run Preprocessing.preprocessing first.")
        # Per-model bundle dirs (sibling, strn-suffixed). The kernel/datain dirs
        # are bundle-relative so the bundle stays relocatable.
        _copytree(src_datain, os.path.join(dest, "datain" + strn))
        os.makedirs(os.path.join(dest, "dataout" + strn), exist_ok=True)
        _copytree(kbuild, os.path.join(dest, "kernels" + strn))
        kdir_subs.append("${CMAKE_CURRENT_SOURCE_DIR}/kernels" + strn)

    tmpl = config.frontend_app_combined_template_dir()
    subs = {
        "EXASIM_VARIANT": variant,
        "MODEL_IDS": ", ".join(str(i) for i in ids),
        "MODEL_ID_LIST": " ".join(str(i) for i in ids),
        "KERNEL_DIRS": " ".join('"' + d + '"' for d in kdir_subs),
    }
    _render(str(tmpl / "CMakeLists.txt.in"),
            os.path.join(dest, "CMakeLists.txt"), subs)
    _render(str(tmpl / "main.cpp.in"),
            os.path.join(dest, "main.cpp"), subs)

    # Source models + a per-model text2code DSL (additive; non-fatal on failure).
    for p in pdes:
        strn = config.model_strn(p)
        modelsrc = str(p.get('modelfile', "")) + ".py"
        if os.path.isfile(modelsrc):
            shutil.copy2(modelsrc, os.path.join(dest, os.path.basename(modelsrc)))
        try:
            from .genpdemodel import genpdemodel
            genpdemodel(p, os.path.join(dest, "pdemodel" + strn + ".txt"))
        except Exception as exc:  # noqa: BLE001 - intentionally non-fatal
            print(f"WARNING: pdemodel{strn}.txt generation skipped ({exc!r}).")

    _write_runscript_combined(dest, pdes, variant, ids)
    _write_manifest_combined(dest, pdes, variant, ids)

    if build:
        _build_and_run_combined(dest, pdes)

    print(f"Exported combined data-transfer app: {dest}")
    return dest


def _combined_pairs(pdes, datain_prefix, dataout_prefix):
    """The N (datain, dataout) argv pairs for a combined run, in slot order."""
    pairs = []
    for p in pdes:
        strn = config.model_strn(p)
        pairs.append(f"{datain_prefix}/datain{strn}/")
        pairs.append(f"{dataout_prefix}/dataout{strn}/out")
    return pairs


def _write_runscript_combined(dest, pdes, variant, ids):
    n = len(pdes)
    pairs = " ".join('"' + s + '"' for s in _combined_pairs(pdes, '$here', '$here'))
    if pdes[0]['mpiprocs'] > 1:
        runline = f'${{MPIRUN:-mpirun}} -np {pdes[0]["mpiprocs"]} ' \
                  f'"$here/build/exasimapp" {n} {pairs}'
    else:
        runline = f'"$here/build/exasimapp" {n} {pairs}'
    text = f"""#!/usr/bin/env bash
# Build and run this exported COMBINED multi-PDE Exasim data-transfer app
# (model ids {", ".join(str(i) for i in ids)}). Requires an Exasim install;
# point EXASIM_ROOT at its prefix. Retarget the variant with EXASIM_VARIANT.
set -euo pipefail
here="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
: "${{EXASIM_ROOT:?Set EXASIM_ROOT to your Exasim install prefix}}"
cmake -S "$here" -B "$here/build" \\
  -DExasim_DIR="$EXASIM_ROOT/lib/cmake/Exasim" \\
  -DEXASIM_VARIANT="${{EXASIM_VARIANT:-{variant}}}"
cmake --build "$here/build" --parallel
{runline}
"""
    path = os.path.join(dest, "run.sh")
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, 0o755)


def _write_manifest_combined(dest, pdes, variant, ids):
    manifest = {
        "format": "exasim-data-transfer-app/1",
        "frontend": "python",
        "combined": True,
        "modelids": ids,
        "numpde": len(pdes),
        "variant": variant,
        "platform": pdes[0]['platform'],
        "mpiprocs": pdes[0]['mpiprocs'],
        "built_against_prefix": str(config.install_prefix()),
    }
    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def _build_and_run_combined(dest, pdes):
    """Configure, build, and run the combined bundle against the local Exasim
    install to verify it before hand-off. Build + outputs go to a scratch dir
    so the shipped bundle stays pristine."""
    print("Verify exported combined app: build + run against the local install...")
    cmake = config.cmake_command()
    work = tempfile.mkdtemp(prefix="exasim_export_verify_combined.")
    try:
        bdir = os.path.join(work, "build")
        subprocess.run([cmake, "-S", dest, "-B", bdir,
                        "-DExasim_DIR=" + str(config.cmake_dir())], check=True)
        jobs = os.environ.get("JOBS") or str(os.cpu_count() or 4)
        subprocess.run([cmake, "--build", bdir, "--parallel", jobs], check=True)
        exe = os.path.join(bdir, "exasimapp")
        if not os.path.exists(exe):
            raise RuntimeError(f"Bundle build did not produce {exe}.")

        n = len(pdes)
        # datain from the (pristine) bundle; dataout into the scratch dir.
        pairs = []
        for p in pdes:
            strn = config.model_strn(p)
            os.makedirs(os.path.join(work, "dataout" + strn), exist_ok=True)
            pairs.append(os.path.join(dest, "datain" + strn) + "/")
            pairs.append(os.path.join(work, "dataout" + strn, "out"))
        if pdes[0]['mpiprocs'] > 1:
            mpirun = pdes[0].get('mpirun') or "mpirun"
            mpitxt = os.path.join(bdir, "mpiexec.txt")
            if os.path.exists(mpitxt):
                discovered = open(mpitxt).read().strip()
                if discovered:
                    mpirun = discovered
            cmd = [mpirun, "-np", str(pdes[0]['mpiprocs']), exe, str(n)] + pairs
        else:
            cmd = [exe, str(n)] + pairs
        subprocess.run(cmd, cwd=work, check=True)

        produced = [f for f in os.listdir(work) if f.startswith("dataout")
                    and os.listdir(os.path.join(work, f))]
        if len(produced) < n:
            raise RuntimeError("Combined bundle verification produced too few outputs.")
        print("Exported combined app verified (built and ran against the local install).")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _build_and_run(dest, pde, numpde):
    """Configure, build, and run the bundle against the local Exasim install
    to verify it works before hand-off.

    The build and the run output go to a throwaway scratch directory so the
    shipped bundle stays pristine (source + datain + empty dataout); the
    scratch only proves the relocatable CMakeLists builds and the solver runs.
    """
    print("Verify exported app: build + run against the local Exasim install...")
    cmake = config.cmake_command()
    work = tempfile.mkdtemp(prefix="exasim_export_verify.")
    try:
        bdir = os.path.join(work, "build")
        subprocess.run([cmake, "-S", dest, "-B", bdir,
                        "-DExasim_DIR=" + str(config.cmake_dir())], check=True)
        jobs = os.environ.get("JOBS") or str(os.cpu_count() or 4)
        subprocess.run([cmake, "--build", bdir, "--parallel", jobs], check=True)

        exe = os.path.join(bdir, "exasimapp")
        if not os.path.exists(exe):
            raise RuntimeError(f"Bundle build did not produce {exe}.")

        datain = os.path.join(dest, "datain") + "/"
        dataout = os.path.join(work, "out")
        if pde['mpiprocs'] > 1:
            mpirun = pde.get('mpirun') or "mpirun"
            mpitxt = os.path.join(bdir, "mpiexec.txt")
            if os.path.exists(mpitxt):
                discovered = open(mpitxt).read().strip()
                if discovered:
                    mpirun = discovered
            cmd = [mpirun, "-np", str(pde['mpiprocs']),
                   exe, str(numpde), datain, dataout]
        else:
            cmd = [exe, str(numpde), datain, dataout]
        subprocess.run(cmd, cwd=work, check=True)

        produced = [f for f in os.listdir(work) if f.startswith("out")]
        if not produced:
            raise RuntimeError("Bundle verification run produced no output.")
        print("Exported app verified (built and ran against the local install).")
    finally:
        shutil.rmtree(work, ignore_errors=True)
