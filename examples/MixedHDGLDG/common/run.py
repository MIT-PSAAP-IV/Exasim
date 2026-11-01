#!/usr/bin/env python3
"""Small examples using native model generation and shared CPU preprocessing.

This is orchestration, not another solver or preprocessor. All generated files
are written below the new /tmp run directory; donor examples stay untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASES = {"Poisson": (2, 1, 2), "Poisson3D": (3, 1, 2),
         "Couette": (2, 4, 7), "ManufacturedNavierStokes": (2, 4, 4),
         "NACA2D": (2, 4, 8)}


def execute(command, log, cwd, timeout=300):
    """Keep exact commands, complete output and failure status, including MPI."""
    started = time.monotonic()
    with log.open("w") as stream:
        stream.write("COMMAND " + json.dumps(list(map(str, command))) + "\n")
        stream.flush()
        process = subprocess.Popen(list(map(str, command)), cwd=cwd,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError("timeout; complete log: " + str(log))
    if status:
        raise RuntimeError("exit %d; complete log: %s" % (status, log))
    return time.monotonic() - started


def setting(text, name, value):
    pattern = r"(?m)^\s*" + re.escape(name) + r"\s*=.*?;\s*$"
    line = name + " = " + value + ";"
    if re.search(pattern, text):
        return re.sub(pattern, lambda _: line, text)
    return text.rstrip() + "\n" + line + "\n"


def grid(case, subdivisions, root):
    # Reuse the existing structured mesh generators. Text2Code's grid format
    # is one-based, unlike the Python native mesh in memory. Convert this
    # example's copy explicitly, without modifying the existing exporter.
    sys.path.insert(0, str(REPO / "frontends/Python"))
    import numpy as np
    from exasim.Mesh.squaremesh import SquareMesh
    from exasim.Mesh.cubemesh import cubemesh
    from exasim.Gencode.exporttext2codemesh import exporttext2codemesh
    if case == "NACA2D":
        shutil.copy2(REPO / "examples/NavierStokes/naca2d/grid.bin", root / "grid.bin")
        return None
    dimension = CASES[case][0]
    p, t = SquareMesh(subdivisions, subdivisions, 1) if dimension == 2 else cubemesh(
        subdivisions, subdivisions, subdivisions, 1)
    exporttext2codemesh({"p": p, "t": t + 1}, root)
    return (np.mean(p[0, t], axis=0) <= .5).astype(int).tolist()


def generation(args, case, target):
    nd, ncu, nparam = CASES[args.case]
    if args.frontend == "python":
        return [args.python, HERE / "generate_python.py", case, target, nd, ncu, nparam]
    if args.frontend == "julia":
        return [args.julia, "--project=" + str(REPO / "frontends/Julia/Exasim"),
                "--compiled-modules=no", HERE / "generate_julia.jl",
                case, target, nd, ncu, nparam]
    # MATLAB's quoting is independent of POSIX shell quoting: no shell runs.
    quote = lambda path: "'" + str(path).replace("'", "''") + "'"
    call = "addpath(%s); generate_matlab(%s,%s,%d,%d,%d)" % (
        quote(HERE), quote(case), quote(target), nd, ncu, nparam)
    return [args.matlab, "-batch", call]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--frontend", required=True, choices=("python", "matlab", "julia", "text2code"))
    parser.add_argument("--scheme", required=True, choices=("hdg", "ldg", "mixed"))
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--h-ranks", type=int, default=1, help="mixed only: world ranks 0..h-1 are H, the rest L")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--exasim-dir", required=True, type=Path)
    parser.add_argument("--preprocess", required=True, type=Path)
    parser.add_argument("--text2code", type=Path)
    parser.add_argument("--model-build", type=Path, help="Reuse this route's already built model; no regeneration")
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--julia", default=str(Path.home() / ".juliaup/bin/julia"))
    parser.add_argument("--matlab", default="/usr/local/bin/matlab")
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/tmp").resolve()) or root.exists():
        parser.error("--root must be a new absolute directory below /tmp")
    # The accuracy cases use p=1,2,3; NACA keeps its donor degree p=4.
    degrees = (4,) if args.case == "NACA2D" else (1, 2, 3)
    if not args.root.is_absolute() or args.N < 2 or args.N % 2 or args.p not in degrees:
        parser.error("use absolute --root, even N >= 2, and p in %s for this case" % (degrees,))
    if args.ranks < 1 or (args.scheme == "mixed" and not 1 <= args.h_ranks < args.ranks):
        parser.error("mixed requires at least one H and one L rank")
    root.mkdir()
    os.environ["EXASIM_DATA_DIR"] = str(REPO / "backend/Preprocessing")
    os.environ["PYTHONPATH"] = str(REPO / "frontends/Python") + os.pathsep + os.environ.get("PYTHONPATH", "")
    case = HERE.parent / args.case
    nd, ncu, nparam = CASES[args.case]
    model = args.model_build.resolve() if args.model_build else root / "model"
    timing = {}
    source = case / ("pdemodel.txt" if args.frontend == "text2code" else
                     "pdemodel." + {"python": "py", "matlab": "m", "julia": "jl"}[args.frontend])
    identity = {"case": args.case, "frontend": args.frontend,
                "model_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    if args.model_build:
        recorded = json.loads((model / "hdgldg_model.json").read_text())
        if any(recorded.get(name) != value for name, value in identity.items()):
            raise RuntimeError("--model-build must match this case, frontend and unchanged model source")
    if not args.model_build:
        if args.frontend != "text2code":
            timing["generate"] = execute(generation(args, case, model), root / "generate.log", root)
        else:
            model.mkdir()
        command = ["cmake", "-S", HERE, "-B", model / "build", "-DCMAKE_BUILD_TYPE=Release",
                   "-DExasim_DIR=" + str(args.exasim_dir.resolve()), "-DEXASIM_SOURCE=" + str(REPO)]
        if args.frontend == "text2code":
            # Text2Code reads pdemodel.txt beside the given pdeapp. Generate
            # kernels only; this route's mesh data comes from the shared step.
            shutil.copy2(case / "pdemodel.txt", model / "pdemodel.txt")
            (model / "pdeapp.txt").write_text(setting(
                (case / "pdeapp.txt").read_text(), "gendatain", "0"))
            command.append("-DPDE_APP=" + str(model / "pdeapp.txt"))
            if args.text2code:
                command.append("-DTEXT2CODE_EXECUTABLE=" + str(args.text2code.resolve()))
        else:
            command.append("-DKERNEL_DIR=" + str(model / "kernels"))
        execute(command, root / "configure.log", root)
        timing["build"] = execute(["cmake", "--build", model / "build", "--parallel", "2"], root / "build.log", root)
    app = model / "build/hdgldg_example"
    if not app.is_file():
        raise RuntimeError("model executable missing: " + str(app))
    executable_hash = hashlib.sha256(app.read_bytes()).hexdigest()
    if args.model_build and recorded.get("executable_sha256") != executable_hash:
        raise RuntimeError("reused model executable changed since its recorded build")
    if not args.model_build:
        (model / "hdgldg_model.json").write_text(json.dumps(
            dict(identity, executable_sha256=executable_hash), indent=2) + "\n")
    manifest = {"case": args.case, "frontend": args.frontend, "scheme": args.scheme,
                "N": args.N, "porder": args.p, "ranks": args.ranks,
                "h_ranks": args.h_ranks if args.scheme == "mixed" else None,
                "model_build": str(model), "executable_sha256": executable_hash}
    (root / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.build_only:
        return 0
    labels = grid(args.case, args.N, root)
    text = (case / "pdeapp.txt").read_text()
    # Save native face records on every physical boundary and on the
    # structural HL code (one past the physical codes) for balance checks.
    codes = re.search(r"(?m)^\s*boundaryconditions\s*=\s*\[([^]]*)\]", text).group(1)
    boundaries = range(1, max(int(code) for code in codes.split(",")) + 2)
    settings = {"porder": str(args.p), "pgauss": str(2 * args.p), "mpiprocs": str(args.ranks),
                "discretization": json.dumps("ldg" if args.scheme == "ldg" else "hdg"),
                "gendatain": "0", "builtinmodelID": "100", "saveSolOpt": "1",
                "datapath": json.dumps(str(root) + "/"), "meshfile": json.dumps("grid.bin"),
                "datainpath": json.dumps(str(root / "datain") + "/"),
                "dataoutpath": json.dumps(str(root / "output") + "/out"),
                "modelfile": json.dumps(str(case / "pdemodel.txt")),
                "ncu": str(ncu), "ncv": "0", "ncw": "0", "nsca": "0", "nvec": "0",
                "nten": "0", "nsurf": "0", "nvqoi": "0", "saveSolBouFreq": "1",
                "ibs": "[" + ",".join(map(str, boundaries)) + "]"}
    for name, value in settings.items():
        text = setting(text, name, value)
    config = root / "pdeapp.txt"
    config.write_text(text)
    if args.scheme == "mixed":
        # Explicit physical split stays x=1/2 as both regions refine. NACA
        # uses a boundary-adjacent region on its one fixed donor mesh instead.
        selection = "selection = labels\nlabels = " + ",".join(map(str, labels)) if labels else (
            "selection = boundary_layers\nseed_boundaries = 1\nlayers = 1")
        spec = root / "selection.txt"
        spec.write_text("version = 1\nworld_size = %d\nh_ranks = %s\nl_ranks = %s\n%s\n" % (
            args.ranks, ",".join(map(str, range(args.h_ranks))),
            ",".join(map(str, range(args.h_ranks, args.ranks))), selection))
        # Native routes state their generated dimensions explicitly; the
        # Text2Code route lets the shared preprocessor parse pdemodel.txt.
        command = [args.preprocess.resolve(), config, spec, root / "prepared"]
        if args.frontend != "text2code":
            command.append("--native-metadata")
        timing["preprocess"] = execute(command, root / "preprocess.log", root)
        command = [args.mpiexec, "-np", args.ranks, app, "--hdgldg", root / "prepared", root / "output"]
    else:
        command = [args.mpiexec, "-np", args.ranks, app, config]
    timing["solve"] = execute(command, root / "solver.log", root)
    manifest["timing_seconds"] = timing
    manifest["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    (root / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Completed", args.case, args.frontend, args.scheme, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
