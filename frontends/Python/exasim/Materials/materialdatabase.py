"""Generic material database DAT/BIN I/O.

External material databases store sampled state/property rows only. They do not
expose any internal material-mesh representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


MATERIAL_DATABASE_MIN_NSTATE = 1
MATERIAL_DATABASE_MAX_NSTATE = 3
MATERIAL_DATABASE_HEADER_SIZE = 5


@dataclass
class MaterialDatabase:
    """Provider-independent structured material sample database."""

    nstate: int
    nprop: int
    dims: tuple[int, int, int]
    rows: np.ndarray

    def __post_init__(self) -> None:
        self.nstate = int(self.nstate)
        self.nprop = int(self.nprop)
        self.dims = tuple(int(x) for x in self.dims)
        self.rows = np.asarray(self.rows, dtype=np.float64)
        validate_material_database(self)
        self.rows = sort_material_database_rows(self)

    @property
    def active_dims(self) -> tuple[int, ...]:
        return self.dims[: self.nstate]

    @property
    def nsamples(self) -> int:
        return int(np.prod(self.active_dims))

    @property
    def state_values(self) -> np.ndarray:
        return self.rows[:, : self.nstate]

    @property
    def property_values(self) -> np.ndarray:
        return self.rows[:, self.nstate :]


def read_material_dat(filename: str | Path) -> MaterialDatabase:
    """Read a text material database."""

    rows = []
    with Path(filename).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].split("%", 1)[0].split("//", 1)[0].strip()
            if line:
                rows.append(np.fromstring(line, sep=" "))
    if len(rows) < 2:
        raise ValueError("material.dat must contain one header row and at least one sample row")
    nstate, nprop, dims = _parse_header(rows[0])
    expected_cols = nstate + nprop
    parsed_rows = rows[1:]
    if any(row.size != expected_cols for row in parsed_rows):
        raise ValueError(f"material.dat sample rows must contain {expected_cols} numeric columns")
    return MaterialDatabase(nstate, nprop, dims, np.asarray(parsed_rows, dtype=np.float64))


def write_material_dat(filename: str | Path, database: MaterialDatabase, *, fmt: str = "%.17g") -> None:
    """Write a text material database with sorted sample rows."""

    database = MaterialDatabase(database.nstate, database.nprop, database.dims, database.rows)
    with Path(filename).open("w", encoding="utf-8") as f:
        f.write(" ".join(f"{x:.17g}" for x in [database.nstate, database.nprop, *database.dims]))
        f.write("\n")
        np.savetxt(f, database.rows, fmt=fmt)


def read_material_database_bin(filename: str | Path) -> MaterialDatabase:
    """Read compact binary material database format.

    The first five values are float64 representations of
    ``nstate nprop n1 n2 n3``. Remaining values are row records.
    """

    data = np.fromfile(filename, dtype=np.float64)
    if data.size < MATERIAL_DATABASE_HEADER_SIZE:
        raise ValueError("material.bin database file is too short")
    nstate, nprop, dims = _parse_header(data[:MATERIAL_DATABASE_HEADER_SIZE])
    nrows = int(np.prod(dims[:nstate]))
    ncols = nstate + nprop
    expected = MATERIAL_DATABASE_HEADER_SIZE + nrows * ncols
    if data.size != expected:
        raise ValueError(f"material.bin contains {data.size} doubles, expected {expected}")
    rows = data[MATERIAL_DATABASE_HEADER_SIZE:].reshape((nrows, ncols))
    return MaterialDatabase(nstate, nprop, dims, rows)


def write_material_database_bin(filename: str | Path, database: MaterialDatabase) -> None:
    """Write compact binary material database format."""

    database = MaterialDatabase(database.nstate, database.nprop, database.dims, database.rows)
    header = np.array([database.nstate, database.nprop, *database.dims], dtype=np.float64)
    np.concatenate((header, database.rows.reshape(-1))).astype(np.float64).tofile(filename)


def validate_material_database(database: MaterialDatabase) -> None:
    """Validate generic material-database structure and completeness."""

    if not (MATERIAL_DATABASE_MIN_NSTATE <= database.nstate <= MATERIAL_DATABASE_MAX_NSTATE):
        raise ValueError("material database requires 1 <= nstate <= 3")
    if database.nprop < 1:
        raise ValueError("material database requires nprop >= 1")
    if len(database.dims) != 3 or any(n <= 0 for n in database.dims):
        raise ValueError("material database requires n1,n2,n3 > 0")
    if database.nstate == 1 and (database.dims[1] != 1 or database.dims[2] != 1):
        raise ValueError("inactive dimensions for nstate=1 require n2=1 and n3=1")
    if database.nstate == 2 and database.dims[2] != 1:
        raise ValueError("inactive dimension for nstate=2 requires n3=1")
    if database.rows.ndim != 2:
        raise ValueError("material database rows must be a 2D array")
    expected_rows = int(np.prod(database.dims[: database.nstate]))
    expected_cols = database.nstate + database.nprop
    if database.rows.shape != (expected_rows, expected_cols):
        raise ValueError(f"material database rows must have shape {(expected_rows, expected_cols)}")
    if not np.all(np.isfinite(database.rows)):
        raise ValueError("material database contains NaN or Inf")
    axes = _unique_axes(database)
    if tuple(len(a) for a in axes) != database.active_dims:
        raise ValueError("material database state coordinates do not match n1,n2,n3")
    keys = [tuple(row[: database.nstate]) for row in database.rows]
    if len(set(keys)) != len(keys):
        raise ValueError("material database contains duplicated state points")
    if len(keys) != int(np.prod([len(a) for a in axes])):
        raise ValueError("material database is missing tensor-product state points")


def sort_material_database_rows(database: MaterialDatabase) -> np.ndarray:
    """Return rows sorted with state_1 varying fastest, then state_2, state_3."""

    validate_shapes_only(database)
    keys = tuple(database.rows[:, i] for i in range(database.nstate))
    order = np.lexsort(keys)
    sorted_rows = database.rows[order, :]
    _validate_complete_sorted_grid(database.nstate, database.dims, sorted_rows)
    return sorted_rows


def _parse_header(header: np.ndarray) -> tuple[int, int, tuple[int, int, int]]:
    if header.size != MATERIAL_DATABASE_HEADER_SIZE:
        raise ValueError("material database header must contain nstate nprop n1 n2 n3")
    if not np.all(np.isfinite(header)):
        raise ValueError("material database header contains NaN or Inf")
    rounded = np.rint(header).astype(int)
    if not np.allclose(header, rounded, rtol=0.0, atol=0.0):
        raise ValueError("material database header entries must be integer-valued")
    return int(rounded[0]), int(rounded[1]), tuple(int(x) for x in rounded[2:5])


def _unique_axes(database: MaterialDatabase) -> tuple[np.ndarray, ...]:
    return tuple(np.unique(database.rows[:, i]) for i in range(database.nstate))


def _validate_complete_sorted_grid(nstate: int, dims: tuple[int, int, int], rows: np.ndarray) -> None:
    axes = tuple(np.unique(rows[:, i]) for i in range(nstate))
    if tuple(len(a) for a in axes) != tuple(dims[:nstate]):
        raise ValueError("material database state coordinates do not match n1,n2,n3")
    expected = []
    for reversed_multi in np.ndindex(tuple(int(n) for n in dims[:nstate][::-1])):
        multi = reversed_multi[::-1]
        expected.append(tuple(axes[i][multi[i]] for i in range(nstate)))
    actual = [tuple(row[:nstate]) for row in rows]
    if actual != expected:
        raise ValueError("material database is missing tensor-product state points")


def validate_shapes_only(database: MaterialDatabase) -> None:
    if database.rows.ndim != 2:
        raise ValueError("material database rows must be a 2D array")
    if database.rows.shape[1] != database.nstate + database.nprop:
        raise ValueError("material database row column count does not match nstate+nprop")
