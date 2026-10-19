"""Python two-stage density-equalizing mesh-adaptivity example."""

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(REPOSITORY / "frontends" / "Python"))

import exasim  # noqa: E402
from exasim.Preprocessing.createdgnodes import createdgnodes  # noqa: E402
from exasim.Preprocessing.facenumbering import facenumbering  # noqa: E402


def _monitor(x, y):
    argument = 100.0 * (x * x + y * y - 0.25**2)
    density = 1.0 + 20.0 / np.cosh(argument)
    scale = -4000.0 * np.sinh(argument) / np.cosh(argument) ** 2
    return density, scale * x, scale * y


def _mean_monitor():
    points, weights = np.polynomial.legendre.leggauss(128)
    xx, yy = np.meshgrid(points, points, indexing="ij")
    values = _monitor(xx, yy)[0]
    return float(np.sum(values * weights[:, None] * weights[None, :]) / 4.0)


def _base_case(run_directory):
    pde, mesh = exasim.initializeexasim()
    pde["model"], pde["platform"], pde["mpiprocs"] = "ModelD", "cpu", 1
    pde["hybrid"], pde["porder"], pde["pgauss"] = 1, 4, 8
    pde["linearsolvertol"], pde["NLtol"] = 1.0e-8, 1.0e-8
    pde["GMRESrestart"], pde["linearsolveriter"] = 200, 400
    pde["preconditioner"], pde["ppdegree"] = 1, 10
    pde["datapath"] = str(run_directory)

    mesh["p"], mesh["t"] = exasim.Mesh.SquareMesh(50, 50, 1)[0:2]
    mesh["p"] = 2.0 * mesh["p"] - 1.0
    mesh["boundaryexpr"] = [
        lambda p: np.abs(p[1, :] + 1.0) < 1.0e-8,
        lambda p: np.abs(p[0, :] - 1.0) < 1.0e-8,
        lambda p: np.abs(p[1, :] - 1.0) < 1.0e-8,
        lambda p: np.abs(p[0, :] + 1.0) < 1.0e-8,
    ]
    mesh["boundarycondition"] = np.ones(4, dtype=np.int64)
    mesh["f"] = facenumbering(mesh["p"], mesh["t"], 1, mesh["boundaryexpr"], [])
    mesh["dgnodes"] = createdgnodes(mesh["p"], mesh["t"], mesh["f"], [], [], pde["porder"])
    return pde, mesh


def main():
    run_directory = HERE / "python_run"
    run_directory.mkdir(exist_ok=True)
    pde, mesh = _base_case(run_directory)
    theta = _mean_monitor()
    x, y = mesh["dgnodes"][:, 0, :], mesh["dgnodes"][:, 1, :]
    density, density_x, density_y = _monitor(x, y)

    pde["modelfile"], pde["physicsparam"], pde["tau"] = "pdemodel_poisson", np.array([1.0, theta]), np.array([2.0])
    pde["builddir"] = str(run_directory / ".exasim" / "poisson")
    mesh["vdg"] = np.stack((density, density_x, density_y), axis=1) / theta
    poisson, _, mesh = exasim.exasim(pde, mesh)[0:3]
    poisson = poisson[..., -1] if poisson.ndim == 4 else poisson

    original = np.array(mesh["dgnodes"], copy=True)
    adapted = np.array(original, copy=True)
    for coordinate in range(2):
        transport, transport_mesh = _base_case(run_directory)
        transport_mesh["dgnodes"] = np.array(original, copy=True)
        transport["modelfile"] = "pdemodel_transport"
        transport["physicsparam"] = np.array([1.0e-4, theta])
        transport["tau"] = np.array([1.0])
        transport["dt"] = 0.02 * np.ones(50)
        transport["soltime"] = np.arange(1, 51, dtype=np.int64)
        transport["torder"], transport["nstage"] = 2, 2
        transport["builddir"] = str(run_directory / ".exasim" / "transport")
        transport_mesh["vdg"] = np.stack((density / theta, density_x / theta,
                                           density_y / theta, poisson[:, 1, :],
                                           poisson[:, 2, :]), axis=1)
        transport_mesh["udg"] = original[:, coordinate:coordinate + 1, :]
        moved = exasim.exasim(transport, transport_mesh)[0]
        adapted[:, coordinate, :] = moved[:, 0, :, -1]

    np.savez(run_directory / "adapted_mesh.npz", dgnodes=adapted)
    print("Python Monge-Ampere square mesh:", adapted.shape)
    return adapted


if __name__ == "__main__":
    main()
