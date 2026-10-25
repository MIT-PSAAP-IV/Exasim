"""Callable driver and artifact writer for the four-domain example."""

import contextlib
import json
import math
import os
import pathlib
import re
import sys
import tempfile

sys.dont_write_bytecode = True

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
CASE = "multiple_interactions_2d"
DIMENSION = 2
FAMILIES = (1, 1, 1, 1)
DEFAULT_PORDER = 1
DEFAULT_N = 2
DEFAULT_RANKS = (1, 1, 1, 1)
FULL_INTERACTION_TABLE = (
    1, 4, 3,
    11, 0, 1, 1, 2, 1, 1, 0, 4, 2, 1, 0,
    11, 1, 2, 1, 2, 2, 1, 0, 4, 1, 1, 0,
    11, 2, 3, 1, 2, 1, 1, 0, 4, 2, 1, 0,
)
SOLVER_CONTROLS = {
    "NLiter": 12,
    "NLtol": 1e-12,
    "linearsolvertol": 1e-12,
    "matvectol": 1e-12,
    "GMRESrestart": 250,
    "linearsolveriter": 600,
}


def _repository_root():
    configured = os.environ.get("EXASIM_ROOT")
    candidates = [pathlib.Path(configured)] if configured else []
    candidates.extend(HERE.parents)
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "frontends" / "Python" / "exasim").is_dir():
            return root
    raise RuntimeError("Cannot locate Exasim; set EXASIM_ROOT to the repository root")


REPOSITORY = _repository_root()
for path in (REPOSITORY / "frontends" / "Python", HERE):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import exasim  # noqa: E402
from exasim.Preprocessing.createdgnodes import createdgnodes  # noqa: E402
from exasim.Preprocessing.facenumbering import facenumbering  # noqa: E402
from exasim.Preprocessing.interactiongraph import interactiongraph  # noqa: E402
from interaction import interaction  # noqa: E402


def _positive_integer(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def parse_ranks(value):
    if isinstance(value, str):
        value = value.split(",")
    ranks = tuple(_positive_integer(int(item), "each domain rank count") for item in value)
    if len(ranks) != 4:
        raise ValueError("ranks must contain exactly four positive integers")
    return ranks


def parse_active_edges(value=(1, 1, 1)):
    if isinstance(value, str):
        value = value.split(",")
    flags = tuple(int(item) for item in value)
    if len(flags) != 3 or any(item not in (0, 1) for item in flags):
        raise ValueError("active_edges must contain exactly three 0/1 values")
    if not any(flags):
        raise ValueError("at least one interaction edge must be active")
    return flags


def prepare_output_directory(outdir=None):
    if outdir is None:
        path = pathlib.Path(tempfile.mkdtemp(prefix=f"exasim-{CASE}-"))
    else:
        raw = pathlib.Path(outdir).expanduser()
        if not raw.is_absolute():
            raise ValueError("outdir must be an absolute path")
        path = raw.resolve()
    if path == REPOSITORY or REPOSITORY in path.parents:
        raise ValueError("outdir must be outside the Exasim repository")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _expected_interaction_table(active_edges):
    values = [1, 4, sum(active_edges)]
    for edge, active in enumerate(active_edges):
        if active:
            start = 3 + 12 * edge
            values.extend(FULL_INTERACTION_TABLE[start:start + 12])
    return np.asarray(values, dtype=np.float64)


def build_case(porder=DEFAULT_PORDER, N=DEFAULT_N, domain_ranks=DEFAULT_RANKS,
               outdir=None, active_edges=(1, 1, 1)):
    """Construct and validate the example without compiling or solving it."""
    porder = _positive_integer(porder, "porder")
    N = _positive_integer(N, "N")
    domain_ranks = parse_ranks(domain_ranks)
    active_edges = parse_active_edges(active_edges)
    run_directory = prepare_output_directory(outdir)

    pde = [None] * 4
    mesh = [None] * 4
    context = dict(
        exasim=exasim, np=np, pathlib=pathlib, pde=pde, mesh=mesh,
        porder=porder, N=N, domainranks=list(domain_ranks),
        families=list(FAMILIES), run_directory=run_directory,
    )
    for filename in ("pdeapp1.py", "pdeapp2.py", "pdeapp3.py", "pdeapp4.py"):
        source = (HERE / filename).read_text(encoding="utf-8")
        exec(compile(source, str(HERE / filename), "exec"), context)

    all_interactions = interaction()
    interactions = [item for item, active in zip(all_interactions, active_edges) if active]
    expected = _expected_interaction_table(active_edges)
    for domain, (app, grid, family, ranks) in enumerate(
            zip(pde, mesh, FAMILIES, domain_ranks), start=1):
        expected_vertices = 4 if family == 1 else 3
        if grid["t"].shape[0] != expected_vertices:
            raise RuntimeError(f"domain {domain} has the wrong element family")
        controls = (app["porder"], app["pgauss"], app["mpiprocs"], app["NLiter"],
                    app["NLtol"], app["linearsolvertol"], app["matvectol"],
                    app["GMRESrestart"], app["linearsolveriter"])
        required = (porder, 2 * porder, ranks, 12, 1e-12, 1e-12, 1e-12, 250, 600)
        if controls != required:
            raise RuntimeError(f"domain {domain} solver controls differ from the canonical case")
        for field in ("datapath", "builddir", "buildpath"):
            if not pathlib.Path(app[field]).resolve().is_relative_to(run_directory):
                raise RuntimeError(f"domain {domain} {field} escapes outdir")
    return dict(
        pde=pde, mesh=mesh, interactions=interactions, table=expected,
        selectors=[[] for _ in range(4)], porder=porder, N=N, domain_ranks=domain_ranks,
        active_edges=active_edges, outdir=run_directory,
    )


@contextlib.contextmanager
def _solver_log(path):
    """Redirect Python and inherited child-process output to ``path``."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with open(path, "w", encoding="utf-8") as stream:
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _global_dgnodes(grid, family, porder):
    faces = facenumbering(
        grid["p"], grid["t"], family, grid["boundaryexpr"], [])[0]
    return createdgnodes(grid["p"], grid["t"], faces, [], [], porder)


def run_example(porder=DEFAULT_PORDER, N=DEFAULT_N, domain_ranks=DEFAULT_RANKS,
                outdir=None, active_edges=(1, 1, 1)):
    """Build, solve, and perform the example's basic correctness checks."""
    case = build_case(porder, N, domain_ranks, outdir, active_edges)
    with _solver_log(case["outdir"] / "solver.log"):
        result = exasim.exasim(case["pde"], case["mesh"], case["interactions"])
    sol, pde, mesh, master, dmd = result[:5]

    log_text = (case["outdir"] / "solver.log").read_text(
        encoding="utf-8", errors="replace")
    residuals = [float(value) for value in re.findall(
        r"Updated Norm:\s*([-+0-9.eE]+)", log_text)]
    if not residuals or not math.isfinite(residuals[-1]):
        raise RuntimeError("solver log contains no finite global Newton residual")
    if residuals[-1] > float(pde[0]["NLtol"]):
        raise RuntimeError("global nonlinear solve did not converge")
    off_face = [int(value) for value in re.findall(
        r"points off donor face = ([0-9]+)", log_text)]
    if len(off_face) < 6 or any(off_face):
        raise RuntimeError("donor-search validation failed")

    max_error = 0.0
    for domain, values in enumerate(sol):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 4:
            values = values[:, :, :, -1]
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"domain {domain + 1} solution is non-finite")
        coordinates = _global_dgnodes(mesh[domain], FAMILIES[domain], case["porder"])
        exact = np.prod(np.sin(np.pi * coordinates), axis=1)
        max_error = max(max_error, float(np.max(np.abs(values[:, 0, :] - exact))))
        mesh[domain]["dgnodes"] = coordinates
    return dict(case=case, sol=sol, pde=pde, mesh=mesh, master=master, dmd=dmd,
                final_residual=residuals[-1], max_error=max_error)
