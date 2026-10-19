"""Python launcher for the canonical reacting Sharp-B Text2Code case."""

from pathlib import Path
import os
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
SOURCE = REPOSITORY / "apps" / "navierstokes" / "reactingsharpb2"


def _prefix():
    candidates = ([Path(os.environ["EXASIM_PREFIX"])] if "EXASIM_PREFIX" in os.environ else []) + [
        REPOSITORY.parent / "exasim_install", REPOSITORY.parent / "Exasim-build" / "install"]
    for candidate in candidates:
        if (candidate / "bin" / "text2code").is_file():
            return candidate
    raise RuntimeError("Set EXASIM_PREFIX to an Exasim installation containing bin/text2code")


def main(run_directory=None):
    run_directory = Path(run_directory or HERE / "python_run")
    if run_directory.exists():
        shutil.rmtree(run_directory)
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE, run_directory,
                    ignore=shutil.ignore_patterns("datain", "dataout", "partition.bin"))
    prefix = _prefix()
    subprocess.run([str(prefix / "bin" / "text2code"), "pdeapp.txt"],
                   cwd=run_directory, check=True)
    subprocess.run(["mpiexec", "-n", "4", str(prefix / "bin" / "exasimapp"), "pdeapp.txt"],
                   cwd=run_directory, check=True)
    print("Python launcher completed:", run_directory / "dataout")
    return run_directory


if __name__ == "__main__":
    main()
