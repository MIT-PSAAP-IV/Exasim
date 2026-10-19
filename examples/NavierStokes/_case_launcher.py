"""Isolated launch support for legacy Navier--Stokes examples.

The launcher never edits the canonical example or application directory.  It
copies the input tree to a frontend-specific run directory, removes stale
solver products, and invokes either Text2Code or MATLAB there.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


def repository_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "CMakeLists.txt").is_file() and (parent / "frontends").is_dir():
            return parent
    raise RuntimeError("could not locate the Exasim repository root")


def exasim_prefix(repository: Path) -> Path:
    configured = os.environ.get("EXASIM_PREFIX")
    candidates = [Path(configured)] if configured else []
    candidates += [
        repository.parent / "exasim_install",
        repository.parent / "Exasim-build" / "install",
    ]
    for candidate in candidates:
        if (candidate / "bin" / "text2code").is_file() and (candidate / "bin" / "exasimapp").is_file():
            return candidate
    raise RuntimeError("set EXASIM_PREFIX to an Exasim installation")


def _fresh_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "datain*", "dataout*", "build", ".exasim", "python_run", "julia_run",
            "__pycache__", "*.pyc",
        ),
    )


def run_text2code(case_file: Path, source: Path, ranks: int, run_directory: Path) -> Path:
    repository = repository_root(case_file)
    _fresh_copy(source, run_directory)
    prefix = exasim_prefix(repository)
    subprocess.run([str(prefix / "bin" / "text2code"), "pdeapp.txt"], cwd=run_directory, check=True)
    subprocess.run(
        ["mpiexec", "-n", str(ranks), str(prefix / "bin" / "exasimapp"), "pdeapp.txt"],
        cwd=run_directory,
        check=True,
    )
    return run_directory / "dataout"


def run_matlab(case_file: Path, source: Path, script: str, run_directory: Path) -> Path:
    _fresh_copy(source, run_directory)
    command = f"cd('{run_directory.as_posix()}'); run('{script}');"
    subprocess.run(["matlab", "-batch", command], check=True)
    return run_directory / "dataout"
