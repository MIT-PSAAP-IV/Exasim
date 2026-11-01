#!/usr/bin/env python3
"""Focused public restart/time-control rejection checks; no solver retuning."""
import argparse
import json
from pathlib import Path
import shutil
import struct

from lifecycle import run


def change_app(path, section, index, value):
    """Change one existing serialized scalar; preserve all sizes and other bytes."""
    raw = bytearray(path.read_bytes())
    count = int(struct.unpack_from("=d", raw)[0])
    sizes = [int(v) for v in struct.unpack_from("=" + "d" * count, raw, 8)]
    if index >= sizes[section]:
        raise ValueError("guard fixture index exceeds native array")
    offset = 8 * (1 + count + sum(sizes[:section]) + index)
    struct.pack_into("=d", raw, offset, value)
    path.write_bytes(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mpiexec", default="mpiexec")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/tmp").resolve()) or root.exists():
        parser.error("--root must be a new /tmp directory")
    root.mkdir(parents=True)
    empty = root / "empty-checkpoint"
    empty.mkdir()
    cases = [
        ("restart-zero", "HDG/LDG restart requires a positive full-timestep number", [], ["--restart", "0", str(empty)]),
        ("restart-negative", "HDG/LDG restart requires a positive full-timestep number", [], ["--restart", "-1", str(empty)]),
        ("restart-fraction", "HDG/LDG restart requires a positive full-timestep number", [], ["--restart", "1.5", str(empty)]),
        ("missing-previous", "HDG/LDG previous output directory is missing", [], ["--restart", "2", str(root / "absent")]),
        ("restart-final", "HDG/LDG time integration supports DIRK22/33 and interior full-step restart only", [], ["--restart", "4", str(empty)]),
        ("missing-checkpoint", "HDG/LDG restart has missing or truncated native checkpoint files", [], ["--restart", "2", str(empty)]),
        ("bdf", "HDG/LDG time integration supports DIRK22/33 and interior full-step restart only", [(None, 2, 2, 1)], []),
        ("stage-order", "HDG/LDG time integration supports DIRK22/33 and interior full-step restart only", [(None, 2, 3, 2)], []),
        ("zero-dt", "HDG/LDG timesteps must be finite and positive", [(None, 4, 0, 0)], []),
        ("different-dt", "HDG/LDG timesteps must agree on all ranks", [("H", 4, 0, .05)], []),
        ("restart-steady", "HDG/LDG checkpoint restart requires a time-dependent run", [(None, 1, 0, 0)], ["--restart", "2", str(empty)]),
        ("runmode", "HDG/LDG public launch requires runmode 0", [(None, 1, 9, 1)], []),
    ]
    # Existing checkpoint filenames with deliberately short payloads exercise
    # the byte-extent rejection independently of missing-path rejection.
    short = root / "short-checkpoint"
    for side in ("H", "L"):
        (short / side).mkdir(parents=True)
        (short / side / "outudg_t2_np0.bin").write_bytes(b"x")
    (short / "H" / "out_uhat_t2_np0.bin").write_bytes(b"x")
    cases.append(("short-checkpoint", "HDG/LDG restart has missing or truncated native checkpoint files", [], ["--restart", "2", str(short)]))
    results = []
    for name, diagnostic, changes, tail in cases:
        case = root / name
        source = case / "input"
        shutil.copytree(args.input, source)
        for side, section, index, value in changes:
            for region in ((side,) if side else ("H", "L")):
                change_app(source / region / "app.bin", section, index, value)
        command = [args.mpiexec, "-np", "2", str(args.app.resolve()), "--hdgldg", str(source), str(case / "output")] + tail
        log = case / "run.log"
        try:
            run(command, log, 30)
        except RuntimeError as error:
            # A timeout, crash without the specific guard, or successful return
            # is not evidence that the intended rejection path worked.
            if not str(error).startswith("command exited ") or diagnostic not in log.read_text():
                raise
        else:
            raise ValueError("unsupported case was accepted: " + name)
        results.append({"name": name, "diagnostic": diagnostic, "command": command})
        print("GUARD PASS " + name, flush=True)
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
