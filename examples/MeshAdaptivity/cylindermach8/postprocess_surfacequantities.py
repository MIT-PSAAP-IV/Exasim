"""Read and plot Mach-8 cylinder wall surfacequantities saved by Exasim."""

from __future__ import annotations

import csv
import glob
import os

import numpy as np

from exasim.Postprocessing.readsurfacequantities import readsurfacequantities


def _count_surface_ranks(prefix: str) -> int:
    return len(glob.glob(prefix + "bouinfo_np*.bin"))


def _fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    c, *_ = np.linalg.lstsq(a, b, rcond=None)
    xc, yc, c0 = c
    radius = np.sqrt(max(c0 + xc * xc + yc * yc, 0.0))
    return float(xc), float(yc), float(radius)


def _average_duplicate_points(*arrays: np.ndarray) -> list[np.ndarray]:
    x = arrays[0]
    y = arrays[1]
    scale = max(float(np.ptp(x)), float(np.ptp(y)), 1.0)
    tol = 1.0e-12 * scale
    keys = np.rint(np.column_stack((x, y)) / tol).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inv).astype(float)
    averaged = []
    for arr in arrays:
        averaged.append(np.bincount(inv, weights=arr) / count)
    return averaged


def _save_csv(filename: str, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    with open(filename, "w", newline="") as fid:
        writer = csv.writer(fid)
        writer.writerow(names)
        for row in zip(*(columns[name] for name in names)):
            writer.writerow([f"{float(value):.17e}" for value in row])


def _try_write_plots(plotdir: str, theta, x, y, cp, cf, cq) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on user environment
        print(f"matplotlib not available; skipping PNG plots ({exc}).")
        return

    for name, values, ylabel, title in [
        ("Cp", cp, "C_p", r"Pressure coefficient, $C_p$"),
        ("Cf", cf, "C_f", r"Skin-friction coefficient, $C_f$"),
        ("Cq", cq, "C_q", r"Heat-flux coefficient, $C_q$"),
    ]:
        fig, ax = plt.subplots()
        ax.plot(theta, values, "o-", linewidth=1.0, markersize=4)
        ax.grid(True)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(plotdir, f"{name}_vs_theta.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots()
        sc = ax.scatter(x, y, c=values, s=24)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{ylabel} on saved wall points")
        fig.colorbar(sc, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(plotdir, f"{name}_wall_points.png"), dpi=200)
        plt.close(fig)


def postprocess_surfacequantities(pde, wallib: int = 3):
    prefix = os.path.join(str(pde["datapath"]), "dataout", "out")
    nranks = int(pde.get("mpiprocs", 1))
    if not os.path.isfile(prefix + f"bouinfo_np{nranks - 1}.bin"):
        nranks = _count_surface_ranks(prefix)
    if nranks < 1:
        raise FileNotFoundError(f"No surface output files found with prefix {prefix!r}")

    surf = readsurfacequantities(prefix, nranks, pde.get("saveSolBouLoc", None))
    if wallib not in surf:
        raise KeyError(f"Boundary ID {wallib} is absent from saved surface output.")
    wall = surf[wallib]
    tstep = wall["values"].shape[3] - 1
    values = wall["values"][:, :, :, tstep]
    xgeo = wall["x"][:, :, :, tstep]
    ngeo = wall["n"][:, :, :, tstep]

    x = xgeo[:, :, 0].reshape(-1, order="F")
    y = xgeo[:, :, 1].reshape(-1, order="F")
    nx = ngeo[:, :, 0].reshape(-1, order="F")
    ny = ngeo[:, :, 1].reshape(-1, order="F")
    cp = values[:, :, 0].reshape(-1, order="F")
    cf = values[:, :, 1].reshape(-1, order="F")
    cq = values[:, :, 2].reshape(-1, order="F")
    x, y, nx, ny, cp, cf, cq = _average_duplicate_points(x, y, nx, ny, cp, cf, cq)

    if not np.all(np.isfinite(np.concatenate((x, y, nx, ny, cp, cf, cq)))):
        raise ValueError("Saved wall coordinates, normals, or quantities contain NaN/Inf.")

    xc, yc, radius = _fit_circle(x, y)
    theta = np.mod(np.arctan2(y - yc, x - xc), 2.0 * np.pi)
    order = np.argsort(theta)
    theta, x, y, nx, ny, cp, cf, cq = [a[order] for a in (theta, x, y, nx, ny, cp, cf, cq)]
    normal_radial_dot = nx * (x - xc) / radius + ny * (y - yc) / radius

    plotdir = os.path.join(str(pde["datapath"]), "surfacequantities_plots")
    os.makedirs(plotdir, exist_ok=True)
    _save_csv(
        os.path.join(plotdir, "surfacequantities.csv"),
        {
            "theta": theta,
            "x": x,
            "y": y,
            "nx": nx,
            "ny": ny,
            "Cp": cp,
            "Cf": cf,
            "Cq": cq,
        },
    )
    _try_write_plots(plotdir, theta, x, y, cp, cf, cq)

    print(f"Surface quantities read from {prefix} with {nranks} rank file(s).")
    print(f"Boundary ID {wallib}, save step {tstep + 1}, {theta.size} unique points.")
    print(f"Fitted center=({xc:.16g}, {yc:.16g}), radius={radius:.16g}.")
    print(
        "normal dot radial range = "
        f"[{normal_radial_dot.min():.6e}, {normal_radial_dot.max():.6e}]."
    )
    print(f"Cp range = [{cp.min():.6e}, {cp.max():.6e}].")
    print(f"Cf range = [{cf.min():.6e}, {cf.max():.6e}].")
    print(f"Cq range = [{cq.min():.6e}, {cq.max():.6e}].")
    print(f"Surface data written to {plotdir}.")
    return {
        "theta": theta,
        "x": np.column_stack((x, y)),
        "n": np.column_stack((nx, ny)),
        "Cp": cp,
        "Cf": cf,
        "Cq": cq,
        "center": (xc, yc),
        "radius": radius,
        "plotdir": plotdir,
    }
