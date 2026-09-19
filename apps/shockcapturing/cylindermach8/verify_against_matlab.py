#!/usr/bin/env python3
"""Compare this text2code run with the MATLAB cylindermach8 reference.

The VTK comparison intentionally reproduces Exasim's VisDG2CG averaging before
comparing with the Float32 point fields stored in outvis.vtu.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
REFERENCE = REPOSITORY / "examples" / "ShockCapturing" / "cylindermach8"
NPE, NC, NCO, NE = 25, 12, 2, 651
GAMMA = 1.4


def read_solution(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    udg = np.fromfile(directory / "outudg_np0.bin", dtype=np.float64)
    if udg.size == 3 + NPE * NC * NE:
        udg = udg[3:]
    udg = udg.reshape((NPE, NC, NE), order="F")

    vdg = np.fromfile(directory / "outvdg_np0.bin", dtype=np.float64)
    if vdg.size == 3 + NPE * NCO * NE:
        vdg = vdg[3:]
    vdg = vdg.reshape((NPE, NCO, NE), order="F")
    return udg, vdg


def fields(udg: np.ndarray, vdg: np.ndarray) -> dict[str, np.ndarray]:
    rho, rhou, rhov, rhoe = (udg[:, i, :] for i in range(4))
    ux = rhou / rho
    uy = rhov / rho
    pressure = (GAMMA - 1.0) * (
        rhoe - 0.5 * (rhou * ux + rhov * uy)
    )
    temperature = pressure / ((GAMMA - 1.0) * rho)
    mach = np.sqrt(ux * ux + uy * uy) / np.sqrt(
        GAMMA * np.abs(pressure) / rho
    )
    av = 0.015 * vdg[:, 1, :] * np.tanh(10.0 * vdg[:, 0, :])
    return {
        "rho": rho,
        "rhou": rhou,
        "rhov": rhov,
        "rhoE": rhoe,
        "u": ux,
        "v": uy,
        "p": pressure,
        "T": temperature,
        "Mach": mach,
        "AV": av,
    }


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    reference_norm = np.linalg.norm(np.ravel(reference))
    return {
        "l2": float(np.linalg.norm(np.ravel(difference))),
        "linf": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(np.ravel(difference)) / reference_norm),
        "reference_min": float(np.min(reference)),
        "reference_max": float(np.max(reference)),
        "text2code_min": float(np.min(actual)),
        "text2code_max": float(np.max(actual)),
    }


def read_vtu_point_data(filename: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    contents = filename.read_bytes()
    appended_tag = contents.index(b'<AppendedData encoding="raw">')
    appended_start = contents.index(b"_", appended_tag) + 1
    header = contents[:appended_tag].decode("ascii")
    point_count = int(re.search(r'NumberOfPoints="(\d+)"', header).group(1))

    arrays = []
    for name, offset in re.findall(
        r'<DataArray type="Float32" Name="([^"]+)"[^>]*offset="(\d+)"/>',
        header,
    ):
        offset = int(offset)
        byte_count = struct.unpack_from("<Q", contents, appended_start + offset)[0]
        value_count = byte_count // np.dtype("<f4").itemsize
        values = np.frombuffer(
            contents,
            dtype="<f4",
            count=value_count,
            offset=appended_start + offset + 8,
        ).copy()
        arrays.append((name, values))

    point_data = {name: values for name, values in arrays if values.size == point_count}
    points = next(values for name, values in arrays if name == "points")
    return points.reshape((-1, 3))[:, :2], point_data


def matlab_fields_on_vtk_points(
    matlab_fields: dict[str, np.ndarray], vtk_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    xdg = np.fromfile(HERE / "xdg.bin", dtype=np.float64)
    if xdg.size == 3 + NPE * 2 * NE:
        xdg = xdg[3:]
    xdg = xdg.reshape((NPE, 2, NE), order="F")

    # Group coincident element nodes, matching VisDG2CG's arithmetic average.
    coordinates = xdg.transpose(2, 0, 1).reshape((-1, 2))
    rounded = np.round(coordinates, decimals=10)
    unique_coordinates, inverse = np.unique(rounded, axis=0, return_inverse=True)
    multiplicity = np.bincount(inverse)

    averaged = []
    for name in ("Mach", "AV"):
        values = matlab_fields[name].transpose(1, 0).reshape(-1)
        averaged.append(np.bincount(inverse, weights=values) / multiplicity)
    averaged = np.column_stack(averaged)

    # VTK stores coordinates as Float32. Match them to the Double-precision CG
    # nodes with a small spatial hash, avoiding any non-NumPy test dependency.
    tolerance = 1.0e-5
    cells: dict[tuple[int, int], list[int]] = {}
    for index, point in enumerate(unique_coordinates):
        key = tuple(np.floor(point / tolerance).astype(np.int64))
        cells.setdefault(key, []).append(index)
    indices = np.empty(len(vtk_points), dtype=np.int64)
    distances = np.empty(len(vtk_points), dtype=np.float64)
    for point_index, point in enumerate(vtk_points):
        key = np.floor(point / tolerance).astype(np.int64)
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(cells.get((key[0] + dx, key[1] + dy), ()))
        if not candidates:
            raise RuntimeError(f"No CG node found near VTK point {point}")
        candidate_points = unique_coordinates[np.asarray(candidates)]
        candidate_distances = np.linalg.norm(candidate_points - point, axis=1)
        local_index = int(np.argmin(candidate_distances))
        indices[point_index] = candidates[local_index]
        distances[point_index] = candidate_distances[local_index]
        if distances[point_index] > tolerance:
            raise RuntimeError(
                f"Nearest CG node is {distances[point_index]} from VTK point {point}"
            )
    if len(np.unique(indices)) != len(vtk_points):
        raise RuntimeError("VTK point-to-CG-node mapping is not one-to-one")
    return averaged[indices, 0], averaged[indices, 1], float(np.max(distances))


def solver_history(filename: Path) -> dict[str, list[float] | list[int]]:
    text = filename.read_text(errors="replace")
    return {
        "initial_residuals": [
            float(value) for value in re.findall(r"Residual Norm: ([0-9.eE+-]+)", text)
        ],
        "updated_residuals": [
            float(value) for value in re.findall(r"Updated Norm: ([0-9.eE+-]+)", text)
        ],
        "gmres_iterations": [
            int(value) for value in re.findall(r"within\s+(\d+) iterations", text)
        ],
    }


def main() -> None:
    matlab_udg, matlab_vdg = read_solution(REFERENCE / "dataout")
    text_udg, text_vdg = read_solution(HERE / "dataout")
    matlab = fields(matlab_udg, matlab_vdg)
    text2code = fields(text_udg, text_vdg)

    result = {
        "dg_fields": {
            name: metrics(matlab[name], text2code[name]) for name in matlab
        }
    }

    vtk_points, vtk_fields = read_vtu_point_data(HERE / "dataout" / "outvis.vtu")
    expected_mach, expected_av, coordinate_error = matlab_fields_on_vtk_points(
        matlab, vtk_points
    )
    result["paraview"] = {
        "field_mapping": {"Scalar Field 0": "Mach", "Scalar Field 1": "AV"},
        "maximum_coordinate_matching_error": coordinate_error,
        "Mach": metrics(expected_mach, vtk_fields["Scalar Field 0"]),
        "AV": metrics(expected_av, vtk_fields["Scalar Field 1"]),
    }

    matlab_history = solver_history(REFERENCE / "matlab_native_run.log")
    text_history = solver_history(HERE / "text2code_run.log")
    updated_difference = np.asarray(text_history["updated_residuals"]) - np.asarray(
        matlab_history["updated_residuals"]
    )
    result["solver_history"] = {
        "matlab_counts": {key: len(value) for key, value in matlab_history.items()},
        "text2code_counts": {key: len(value) for key, value in text_history.items()},
        "initial_residuals_identical": (
            matlab_history["initial_residuals"] == text_history["initial_residuals"]
        ),
        "gmres_iterations_identical": (
            matlab_history["gmres_iterations"] == text_history["gmres_iterations"]
        ),
        "maximum_printed_updated_residual_difference": float(
            np.max(np.abs(updated_difference))
        ),
    }

    report = HERE / "verification_results.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
