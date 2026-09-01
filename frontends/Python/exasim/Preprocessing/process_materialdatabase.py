"""Stage optional material databases during frontend preprocessing."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np


def process_materialdatabase(pde, output_dir, output_name="materialdatabase.bin"):
    """Stage ``pde.materialdatabase`` under ``output_dir`` when it is set.

    Empty paths are a no-op. ``.bin`` files are copied byte-for-byte. ``.dat``
    files are converted to Exasim's standard raw Float64 binary layout while
    preserving numeric values in file order.
    """

    source_value = pde.get("materialdatabase", "") if isinstance(pde, dict) else getattr(pde, "materialdatabase", "")
    source_text = str(source_value).strip()
    if not source_text:
        return None

    source = Path(source_text)
    if not source.is_file():
        raise FileNotFoundError(f"pde.materialdatabase file not found: {source}")

    destination = Path(output_dir) / str(output_name)
    suffix = source.suffix.lower()
    if suffix == ".bin":
        shutil.copyfile(source, destination)
    elif suffix == ".dat":
        values = _read_materialdatabase_dat_values(source)
        values.astype(np.float64).tofile(destination)
    else:
        raise ValueError(f"Unsupported pde.materialdatabase format '{suffix}'. Expected .dat or .bin.")
    return destination


def _read_materialdatabase_dat_values(filename: Path) -> np.ndarray:
    values = []
    comment_re = re.compile(r"(#|%|//).*$")
    with filename.open("r", encoding="utf-8") as f:
        for line in f:
            line = comment_re.sub("", line).strip()
            if not line:
                continue
            values.extend(float(token) for token in line.split())
    return np.asarray(values, dtype=np.float64)
