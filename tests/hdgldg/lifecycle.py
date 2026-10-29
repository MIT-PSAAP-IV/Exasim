#!/usr/bin/env python3
"""Small public DIRK/restart comparison; all generated files stay under /tmp.

Preparation is intentionally separate: the caller supplies the fixture's native
inputs, so this checker does not implement another mesh or input writer.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess


STAGE = re.compile(r"Timestep\s*:\s*(\d+),\s*Timestage\s*:\s*(\d+)")
# Only numerical solver records are compared. Native elapsed-time/profiling
# lines and filesystem paths are not reproducibility evidence.
HISTORY = re.compile(
    r"^(?:HDGLDG (?:initial residual|Newton )|Newton [Ii]teration:|"
    r"\s*GMRES(?:\([0-9]+\))? finished|"
    r"\s*(?:Nonlinear|nonlinear|Linear|linear) (?:residual|iteration))"
)
RESIDUAL = re.compile(r"(?:initial residual | residual |Residual Norm: |Updated Norm: )([^ ,]+)")


def run(command, log, timeout):
    """Kill the whole MPI process group on timeout, not just mpiexec."""
    with log.open("w") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError("timeout; MPI process group stopped: " + str(log))
    if code:
        raise RuntimeError("command exited %d; see %s" % (code, log))


def history(log, first, last, order, step_offset=0, nonlinear_tolerance=1e-6):
    result = {}
    current = None
    for line in log.read_text().splitlines():
        stage = STAGE.search(line)
        if stage:
            step, substage = map(int, stage.groups())
            current = (step + step_offset, substage)
            if current in result:
                raise ValueError("duplicate stage %s in %s" % (current, log))
            result[current] = []
        elif current is not None and HISTORY.search(line):
            result[current].append(line.strip())
    expected = [(step, stage) for step in range(first, last + 1)
                for stage in range(1, order + 1)]
    if list(result) != expected:
        raise ValueError("incomplete or reordered stage sequence in %s: %s" % (log, list(result)))
    if any(not records for records in result.values()):
        raise ValueError("stage without numerical solver records in " + str(log))
    for stage, records in result.items():
        residuals = [float(match.group(1)) for line in records
                     for match in RESIDUAL.finditer(line)]
        if not residuals or not math.isfinite(residuals[-1]) or residuals[-1] > nonlinear_tolerance:
            raise ValueError("stage %s did not meet the configured nonlinear tolerance in %s" % (stage, log))
    return result


def compare_file(reference, replay):
    a, b = reference.read_bytes(), replay.read_bytes()
    if not a or len(a) != len(b) or len(a) % 8:
        raise ValueError("invalid native checkpoint extents: %s, %s" % (reference, replay))
    av = struct.unpack("=" + "d" * (len(a) // 8), a)
    bv = struct.unpack("=" + "d" * (len(b) // 8), b)
    if not all(math.isfinite(v) for v in av + bv):
        raise ValueError("nonfinite checkpoint: " + str(reference))
    maximum = max(abs(x - y) for x, y in zip(av, bv))
    scale = max(1., max(map(abs, av)), max(map(abs, bv)))
    return {"name": reference.name, "bytes": len(a), "bitwise": a == b,
            "max_absolute": maximum, "max_scaled": maximum / scale,
            "reference_sha256": hashlib.sha256(a).hexdigest(),
            "replay_sha256": hashlib.sha256(b).hexdigest()}


def checkpoint_files(output, step):
    # Native checkpoint names deliberately differ from streamed outudg.bin.
    return sorted(output.rglob("outudg_t%d_np*.bin" % step)) + sorted(
        output.rglob("out_uhat_t%d_np*.bin" % step))


def input_manifest(directory):
    return {str(file.relative_to(directory)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(directory.rglob("*")) if file.is_file()}


def region_rank_count(input_root, side):
    tokens = (input_root / side / "hdgldg.meta").read_text().split()
    if tokens[:2] != ["EXASIM_HDGLDG_REGION", "1"]:
        raise ValueError("unsupported region metadata")
    # scheme, world-rank count, endpoint count, boundary code, mesh dimension.
    return int(tokens[3])


def native_config(path, datain, dataout, ranks, scheme):
    # gendatain=1 is the existing native binary-input route, not Text2Code.
    def quote(value):
        return json.dumps(str(value))
    path.write_text('model = "ModelD";\nplatform = "cpu";\ngendatain = 1;\n'
                    'modelfile = "unused.txt";\nmeshfile = "unused.bin";\n'
                    'discretization = "%s";\nmpiprocs = %d;\nporder = 2;\npgauss = 6;\n'
                    'physicsparam = [];\ntau = [1];\nboundaryconditions = [1,2,3,4];\n'
                    'boundaryexpressions = ["abs(y)<1e-12","abs(x-1)<1e-12",'
                    '"abs(y-1)<1e-12","abs(x)<1e-12"];\n'
                    'builtinmodelID = 0;\nncu = 3;\nncv = 0;\nncw = 0;\n'
                    'nsca = 0;\nnvec = 0;\nnten = 0;\nnsurf = 0;\nnvqoi = 0;\n'
                    'datainpath = %s;\ndataoutpath = %s;\n' %
                    ("hdg" if scheme == "H" else "ldg", ranks, quote(datain), quote(dataout)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--np-flag", default="-np")
    parser.add_argument("--ranks", required=True, type=int)
    parser.add_argument("--order", required=True, type=int, choices=(2, 3))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pure-scheme", choices=("H", "L"))
    parser.add_argument("--pure-resume-input", type=Path)
    parser.add_argument("--timeout", default=120, type=float)
    parser.add_argument("--nonlinear-tolerance", default=1e-6, type=float,
                        help="existing fixture tolerance; does not change solver inputs")
    parser.add_argument("--diagnostic", action="store_true",
                        help="measure a pure replay baseline without assuming bitwise equality")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/tmp").resolve()) or root.exists():
        parser.error("--root must be a new absolute directory under /tmp")
    if bool(args.pure_scheme) != bool(args.pure_resume_input):
        parser.error("pure replay needs --pure-scheme and --pure-resume-input together")
    root.mkdir(parents=True)
    original_inputs = input_manifest(args.input)
    original_resume_inputs = input_manifest(args.pure_resume_input) if args.pure_scheme else {}
    full, resumed = root / "full", root / "resumed"
    mpi = [args.mpiexec, args.np_flag, str(args.ranks), str(args.app.resolve())]
    input_hashes = {}
    if args.pure_scheme:
        # The native writer is called twice only to change dt/timestepOffset.
        # Mesh, master and initial full-nc solution arrays must not drift.
        inputs = sorted(args.input.glob("*.bin"))
        if not inputs:
            raise ValueError("pure native inputs are absent")
        for file in inputs:
            if file.name == "app.bin":
                continue
            original = file.read_bytes()
            if original != (args.pure_resume_input / file.name).read_bytes():
                raise ValueError("pure restart input changed mesh/master/initial state: " + file.name)
            input_hashes[file.name] = hashlib.sha256(original).hexdigest()
        full.mkdir(); resumed.mkdir()
        native_config(root / "full.txt", args.input.resolve(), full, args.ranks, args.pure_scheme)
        native_config(root / "resumed.txt", args.pure_resume_input.resolve(), resumed,
                      args.ranks, args.pure_scheme)
        full_command = mpi + [str(root / "full.txt")]
    else:
        full_command = mpi + ["--hdgldg", str(args.input.resolve()), str(full)]
    run(full_command, root / "full.log", args.timeout)
    if args.pure_scheme:
        checkpoints = checkpoint_files(full, 2)
        expected = args.ranks * (2 if args.pure_scheme == "H" else 1)
        if len(checkpoints) != expected:
            raise ValueError("wrong pure step-2 checkpoint count")
        for file in checkpoints:
            shutil.copyfile(file, resumed / file.name)
        resume_command = mpi + [str(root / "resumed.txt")]
    else:
        expected = args.ranks + region_rank_count(args.input, "H")
        resume_command = mpi + ["--hdgldg", str(args.input.resolve()), str(resumed),
                                "--restart", "2", str(full)]
    run(resume_command, root / "resumed.log", args.timeout)
    full_history = history(root / "full.log", 1, 4, args.order,
                           nonlinear_tolerance=args.nonlinear_tolerance)
    # The native pure driver prints run-relative step numbers even though its
    # checkpoint filenames include timestepOffset. Coupled logs are absolute.
    resumed_history = history(root / "resumed.log", 3, 4, args.order,
                              step_offset=2 if args.pure_scheme else 0,
                              nonlinear_tolerance=args.nonlinear_tolerance)
    histories_equal = all(full_history[key] == value for key, value in resumed_history.items())
    final = checkpoint_files(full, 4)
    if len(final) != expected:
        raise ValueError("wrong final checkpoint count: %d != %d" % (len(final), expected))
    files = []
    for file in final:
        measurement = compare_file(file, resumed / file.relative_to(full))
        measurement["name"] = str(file.relative_to(full))
        files.append(measurement)
    if len(checkpoint_files(resumed, 4)) != expected:
        raise ValueError("wrong resumed final checkpoint count")
    if input_manifest(args.input) != original_inputs or (args.pure_scheme and
            input_manifest(args.pure_resume_input) != original_resume_inputs):
        raise ValueError("application changed prepared input files")
    result = {"order": args.order, "ranks": args.ranks, "pure_scheme": args.pure_scheme,
              "dt": 1 / 40, "final_time": .1, "steps": 4, "restart_step": 2,
              "nonlinear_tolerance": args.nonlinear_tolerance,
              "history_bitwise": histories_equal,
              "fields_bitwise": all(file["bitwise"] for file in files),
              "pure_mesh_master_initial_sha256": input_hashes,
              "prepared_input_sha256": original_inputs,
              "prepared_resume_input_sha256": original_resume_inputs,
              "full_history": {"%d:%d" % key: value for key, value in full_history.items()},
              "resumed_history": {"%d:%d" % key: value for key, value in resumed_history.items()},
              "files": files, "commands": [full_command, resume_command]}
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    exact = histories_equal and result["fields_bitwise"]
    print("LIFECYCLE %s order=%d np=%d fields_bitwise=%s history_bitwise=%s" %
          ("PASS" if exact else "DIFFERENT", args.order, args.ranks,
           result["fields_bitwise"], histories_equal))
    return 0 if exact or args.diagnostic else 1


if __name__ == "__main__":
    raise SystemExit(main())
