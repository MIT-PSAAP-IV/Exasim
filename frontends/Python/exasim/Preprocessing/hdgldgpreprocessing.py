"""Shared CPU preprocessing only; native model generation remains unchanged."""

from pathlib import Path
import subprocess


def hdgldgpreprocessing(config, manifest, outdir, executable):
    """Return (H_datain, L_datain) from explicit metadata and selection files.

    ``config`` is an existing backend configuration with explicit model sizes
    (including ncu/ncv/ncw), mesh/field files and native solver controls.
    ``outdir`` must be a new absolute directory. This does not generate model
    kernels, run a solver, or change native pure preprocessing.
    """
    config = Path(config).resolve(strict=True)
    manifest = Path(manifest).resolve(strict=True)
    executable = Path(executable).resolve(strict=True)
    output = Path(outdir)
    if not output.is_absolute() or output.exists():
        raise ValueError("rank-separated output must be a new absolute directory")
    for name, path in (("config", config), ("manifest", manifest), ("executable", executable)):
        if not path.is_file():
            raise ValueError("rank-separated " + name + " must be a file")
    subprocess.run([str(executable), str(config), str(manifest), str(output),
                    "--native-metadata"], check=True)
    if not (output / "selection.txt").is_file():
        raise RuntimeError("rank-separated preprocessing omitted selection.txt")
    directories = [output / "H", output / "L"]
    for expected_scheme, directory in zip((1, 0), directories):
        metadata = directory / "hdgldg.meta"
        with metadata.open(encoding="utf-8") as stream:
            if stream.readline().strip() != "EXASIM_HDGLDG_REGION 1":
                raise RuntimeError("rank-separated preprocessing produced invalid metadata")
            values = stream.readline().split()
        if len(values) < 2 or int(values[0]) != expected_scheme or int(values[1]) < 1:
            raise RuntimeError("rank-separated preprocessing produced invalid region sizes")
        count = int(values[1])
        required = ["app.bin", "master.bin"]
        for rank in range(count):
            suffix = str(rank + 1) if count > 1 else ""
            required.extend(("mesh" + suffix + ".bin", "sol" + suffix + ".bin",
                             "hdgldg" + suffix + ".txt"))
        for filename in required:
            if not (directory / filename).is_file():
                raise RuntimeError("rank-separated preprocessing omitted " + str(directory / filename))
    return tuple(str(directory) for directory in directories)
