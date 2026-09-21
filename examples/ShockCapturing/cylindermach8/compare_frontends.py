"""Compare the MATLAB, Python, and Julia Mach-8 cylinder runs.

Run pdeapp.m, pdeapp.py, and pdeapp.jl first.  This script checks the
serialized Exasim inputs, compares conservative and derived output fields,
and writes a common three-frontend visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
NPE = 25
NC = 12
NCO = 2
NE = 651
GAMMA = 1.4


@dataclass
class Run:
    name: str
    root: Path
    udg: np.ndarray
    vdg: np.ndarray


def read_doubles(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run all three frontends first.")
    return np.fromfile(path, dtype=np.float64)


def read_output(root: Path, stem: str, shape: tuple[int, ...]) -> np.ndarray:
    values = read_doubles(root / "dataout" / f"{stem}_np0.bin")
    expected = int(np.prod(shape))
    if values.size == expected + len(shape):
        header = tuple(int(round(value)) for value in values[: len(shape)])
        if header != shape:
            raise ValueError(f"Unexpected {stem} header {header}; expected {shape}.")
        values = values[len(shape) :]
    if values.size != expected:
        raise ValueError(
            f"{stem} in {root} has {values.size} entries; expected {expected}."
        )
    return values.reshape(shape, order="F")


def read_structured_file(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    values = read_doubles(path)
    if values.size < 2:
        raise ValueError(f"Malformed Exasim binary file: {path}")
    ncount = int(round(values[0]))
    nsize = values[1 : 1 + ncount].astype(np.int64)
    offset = 1 + ncount
    fields: list[np.ndarray] = []
    for count in nsize:
        end = offset + int(count)
        if end > values.size:
            raise ValueError(f"Malformed size table in {path}")
        fields.append(values[offset:end])
        offset = end
    if offset != values.size:
        raise ValueError(f"{path} contains {values.size - offset} trailing values")
    return nsize, fields


def norms(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    error = np.asarray(candidate) - np.asarray(reference)
    if error.size == 0:
        return 0.0, 0.0, 0.0
    l2 = float(np.linalg.norm(error.ravel(order="F")))
    linf = float(np.max(np.abs(error)))
    denominator = float(np.linalg.norm(np.asarray(reference).ravel(order="F")))
    relative = l2 / denominator if denominator > 0.0 else l2
    return l2, linf, relative


def derived_fields(run: Run) -> dict[str, np.ndarray]:
    rho, rhou, rhov, rhoe = (run.udg[:, i, :] for i in range(4))
    velocity_x = rhou / rho
    velocity_y = rhov / rho
    pressure = (GAMMA - 1.0) * (
        rhoe - 0.5 * (rhou * rhou + rhov * rhov) / rho
    )
    temperature = pressure / ((GAMMA - 1.0) * rho)
    sound_speed = np.sqrt(GAMMA * pressure / rho)
    mach = np.sqrt(velocity_x * velocity_x + velocity_y * velocity_y) / sound_speed
    wall_distance = run.vdg[:, 0, :]
    sensor = run.vdg[:, 1, :]
    effective_av = (0.060 + 0.015 * sensor) * np.tanh(10.0 * wall_distance)
    return {
        "rho": rho,
        "rhou": rhou,
        "rhov": rhov,
        "rhoE": rhoe,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "pressure": pressure,
        "temperature": temperature,
        "Mach": mach,
        "wall_distance": wall_distance,
        "AV_sensor": sensor,
        "effective_AV": effective_av,
    }


def compare_inputs(runs: list[Run]) -> list[str]:
    lines = ["SERIALIZED INPUT COMPARISON"]
    pairs = ((0, 1), (0, 2), (1, 2))

    # app.bin is a field-based file.  Report semantic fields rather than
    # requiring byte identity from independently evaluated floating arithmetic.
    app_names = [
        "ndims",
        "flag",
        "problem",
        "externalparam",
        "dt",
        "factor",
        "physicsparam",
        "solversparam",
        "tau",
        "stgdata",
        "stgparam",
        "stgib",
        "vindx",
        "dae_dt",
        "interfacefluxmap",
        "avparam",
        "wmModelIDs",
        "wmBoundaries",
        "wmDistances",
        "avfilterparam",
    ]
    app_data = {}
    for run in runs:
        nsize, fields = read_structured_file(run.root / "datain" / "app.bin")
        app_data[run.name] = (nsize, fields)
        lines.append(f"  {run.name:6s} app nsize = {nsize.tolist()}")
    for ia, ib in pairs:
        a, b = runs[ia], runs[ib]
        nsa, fa = app_data[a.name]
        nsb, fb = app_data[b.name]
        if not np.array_equal(nsa, nsb):
            raise AssertionError(f"app.bin nsize differs for {a.name} and {b.name}")
        worst = ("", 0.0, 0.0, 0.0)
        for index, (xa, xb) in enumerate(zip(fa, fb)):
            metric = norms(xa, xb)
            if metric[1] > worst[2]:
                label = app_names[index] if index < len(app_names) else f"field_{index}"
                worst = (label, *metric)
        lines.append(
            f"  app {a.name:6s} vs {b.name:6s}: worst={worst[0]}, "
            f"L2={worst[1]:.6e}, Linf={worst[2]:.6e}, relL2={worst[3]:.6e}"
        )

    # mesh.bin must be exactly equal after translation to zero-based storage.
    mesh_values = {run.name: read_doubles(run.root / "datain" / "mesh.bin") for run in runs}
    for ia, ib in pairs:
        a, b = runs[ia], runs[ib]
        metric = norms(mesh_values[a.name], mesh_values[b.name])
        lines.append(
            f"  mesh {a.name:6s} vs {b.name:6s}: "
            f"L2={metric[0]:.6e}, Linf={metric[1]:.6e}, relL2={metric[2]:.6e}"
        )

    # The first five sol.bin payloads are ndims, xdg, initial udg, initial vdg,
    # and wdg.  Their ordering is shared by all frontends.
    sol_names = ("ndims", "xdg", "initial_udg", "initial_vdg", "wdg")
    sol_data = {
        run.name: read_structured_file(run.root / "datain" / "sol.bin") for run in runs
    }
    for ia, ib in pairs:
        a, b = runs[ia], runs[ib]
        nsa, fa = sol_data[a.name]
        nsb, fb = sol_data[b.name]
        if not np.array_equal(nsa, nsb):
            raise AssertionError(f"sol.bin nsize differs for {a.name} and {b.name}")
        # MATLAB records partition/face counts in sol.ndims, whereas the Python
        # and Julia writers record component counts.  The backend sizes and
        # reads the payload from nsize/app/master, so compare the actual fields
        # and document (rather than mislabel) this harmless metadata difference.
        lines.append(
            f"  sol/ndims metadata {a.name:6s}={fa[0].astype(int).tolist()} "
            f"{b.name:6s}={fb[0].astype(int).tolist()}"
        )
        for label, xa, xb in zip(sol_names[1:], fa[1:], fb[1:]):
            metric = norms(xa, xb)
            lines.append(
                f"  sol/{label:11s} {a.name:6s} vs {b.name:6s}: "
                f"L2={metric[0]:.6e}, Linf={metric[1]:.6e}, relL2={metric[2]:.6e}"
            )

    master_values = {
        run.name: read_doubles(run.root / "datain" / "master.bin") for run in runs
    }
    for ia, ib in pairs:
        a, b = runs[ia], runs[ib]
        metric = norms(master_values[a.name], master_values[b.name])
        lines.append(
            f"  master {a.name:6s} vs {b.name:6s}: "
            f"L2={metric[0]:.6e}, Linf={metric[1]:.6e}, relL2={metric[2]:.6e}"
        )
    return lines


def compare_outputs(runs: list[Run]) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    fields = {run.name: derived_fields(run) for run in runs}
    lines = ["", "FINAL-SOLUTION COMPARISON"]
    pairs = ((0, 1), (0, 2), (1, 2))
    for ia, ib in pairs:
        a, b = runs[ia], runs[ib]
        lines.append(f"  {a.name} vs {b.name}")
        lines.append("    field             L2                 Linf               relative L2")
        for label in fields[a.name]:
            metric = norms(fields[a.name][label], fields[b.name][label])
            lines.append(
                f"    {label:14s} {metric[0]:.9e}  {metric[1]:.9e}  {metric[2]:.9e}"
            )
    return lines, fields


def coordinates_from_sol(root: Path) -> np.ndarray:
    _, fields = read_structured_file(root / "datain" / "sol.bin")
    return fields[1].reshape((NPE, 2, NE), order="F")


def make_plot(
    runs: list[Run], fields: dict[str, dict[str, np.ndarray]], output: Path
) -> None:
    xdg = coordinates_from_sol(runs[0].root)
    x = xdg[:, 0, :].ravel(order="F")
    y = xdg[:, 1, :].ravel(order="F")
    plot_fields = (
        ("Mach", "Mach number"),
        ("AV_sensor", "Filtered AV sensor"),
        ("effective_AV", "Effective artificial viscosity"),
    )
    figure = plt.figure(figsize=(12, 12), constrained_layout=True)
    grid = figure.add_gridspec(
        len(plot_fields), len(runs) + 1, width_ratios=(1.0, 1.0, 1.0, 0.05)
    )
    axes = np.empty((len(plot_fields), len(runs)), dtype=object)
    for row, (key, title) in enumerate(plot_fields):
        all_values = np.concatenate([fields[run.name][key].ravel() for run in runs])
        value_min = float(np.min(all_values))
        value_max = float(np.max(all_values))
        artist = None
        for column, run in enumerate(runs):
            ax = figure.add_subplot(grid[row, column])
            axes[row, column] = ax
            values = fields[run.name][key].ravel(order="F")
            artist = ax.scatter(
                x, y, c=values, s=1.5, linewidths=0, vmin=value_min, vmax=value_max
            )
            ax.set_aspect("equal")
            ax.set_xlim(-3.0, 0.05)
            ax.set_ylim(-4.8, 4.8)
            ax.set_title(f"{run.name}: {title}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        color_axis = figure.add_subplot(grid[row, -1])
        figure.colorbar(artist, cax=color_axis)
    figure.suptitle("Mach-8 cylinder: MATLAB/Python/Julia frontend comparison")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    roots = {
        "MATLAB": HERE,
        "Python": HERE / "python_run",
        "Julia": HERE / "julia_run",
    }
    runs = [
        Run(
            name,
            root,
            read_output(root, "outudg", (NPE, NC, NE)),
            read_output(root, "outvdg", (NPE, NCO, NE)),
        )
        for name, root in roots.items()
    ]
    lines = compare_inputs(runs)
    output_lines, fields = compare_outputs(runs)
    lines.extend(output_lines)
    report = HERE / "frontend_comparison.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    image = HERE / "frontend_comparison.png"
    make_plot(runs, fields, image)
    print("\n".join(lines))
    print(f"\nWrote {report}")
    print(f"Wrote {image}")


if __name__ == "__main__":
    main()
