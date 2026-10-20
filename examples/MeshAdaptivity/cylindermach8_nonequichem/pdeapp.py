"""Python launcher for the canonical finite-rate cylinder Text2Code app."""

from pathlib import Path
import os
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
SOURCE = REPOSITORY / "apps" / "meshadaptivity" / "cylindermach8_nonequichem"


def _prefix():
    configured = os.environ.get("EXASIM_PREFIX")
    candidates = [Path(configured)] if configured else []
    candidates += [REPOSITORY.parent / "exasim_install", REPOSITORY.parent / "Exasim-build" / "install"]
    for candidate in candidates:
        if (candidate / "bin" / "text2code").is_file():
            return candidate
    raise RuntimeError("Set EXASIM_PREFIX to an Exasim installation containing bin/text2code")


def main(run_directory=None):
    run_directory = Path(run_directory or HERE / "python_run")
    if run_directory.exists():
        shutil.rmtree(run_directory)
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE, run_directory, ignore=shutil.ignore_patterns("datain", "dataout"))
    prefix = _prefix()
    subprocess.run([str(prefix / "bin" / "text2code"), "pdeapp.txt"], cwd=run_directory, check=True)
    subprocess.run(["mpiexec", "-n", "8", str(prefix / "bin" / "exasimapp"), "pdeapp.txt"], cwd=run_directory, check=True)
    print("Python launcher completed:", run_directory / "dataout")
    return run_directory


if __name__ == "__main__":
    main()
