"""Material database utilities for Exasim Python frontends.

This package intentionally exposes only the external material database
representation used by frontend preprocessing and export workflows.
"""

from .materialdatabase import (
    MaterialDatabase,
    read_material_database_bin,
    read_material_dat,
    sort_material_database_rows,
    validate_material_database,
    write_material_database_bin,
    write_material_dat,
)

__all__ = [
    "MaterialDatabase",
    "read_material_database_bin",
    "read_material_dat",
    "sort_material_database_rows",
    "validate_material_database",
    "write_material_database_bin",
    "write_material_dat",
]
