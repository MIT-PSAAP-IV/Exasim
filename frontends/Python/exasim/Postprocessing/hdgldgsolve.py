"""Native Python model generation and shared rank-separated preprocessing."""
import copy
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .. import config, Gencode, Preprocessing
from ..Gencode.exporttext2code import _infer_dimensions


def _collect_solution(datain, dataout, ne, return_schemes=False):
    """Gather owned output using explicit input-element IDs in the sidecars."""
    solution, assigned = None, np.zeros(ne, dtype=bool)
    schemes = np.zeros(ne, dtype=np.int64)
    for side in ("H", "L"):
        region = datain / side
        metadata = (region / "hdgldg.meta").read_text().split()
        if metadata[:2] != ["EXASIM_HDGLDG_REGION", "1"]:
            raise RuntimeError("unsupported HDG/LDG region metadata")
        ranks = int(metadata[3])
        for rank in range(ranks):
            suffix = str(rank + 1) if ranks > 1 else ""
            records = (region / ("hdgldg" + suffix + ".txt")).read_text().split()
            if records[:2] != ["EXASIM_HDGLDG", "1"]:
                raise RuntimeError("unsupported HDG/LDG rank metadata")
            world, local, count = int(records[3]), int(records[4]), int(records[7])
            entries = np.asarray(records[8:8 + 3 * count], dtype=np.int64).reshape(count, 3)
            if local != rank or not np.array_equal(entries[:, 0], np.arange(count)):
                raise RuntimeError("invalid HDG/LDG local element ordering")
            owned_indices = np.flatnonzero(entries[:, 2] == world)
            if not np.array_equal(owned_indices, np.arange(len(owned_indices))):
                raise RuntimeError("native owned elements must precede ghosts")
            ids = entries[owned_indices, 1]
            output = np.fromfile(dataout / side / ("outudg_np" + str(rank) + ".bin"), dtype=np.float64)
            if len(output) < 3 or np.any(output[:3] != output[:3].astype(np.int64)):
                raise RuntimeError("invalid native solution header")
            npe, nc, owned = output[:3].astype(np.int64)
            size = npe * nc * owned
            if npe <= 0 or nc <= 0 or owned != len(ids) or size <= 0 or (len(output) - 3) % size:
                raise RuntimeError("incomplete native HDG/LDG solution")
            if len(output) < 3 + size or not np.all(np.isfinite(output[-size:])):
                raise RuntimeError("missing or nonfinite native HDG/LDG solution")
            if np.any(ids < 0) or np.any(ids >= ne) or len(np.unique(ids)) != len(ids) or np.any(assigned[ids]):
                raise RuntimeError("duplicate or invalid owned original element ID")
            if solution is None:
                solution = np.empty((npe, nc, ne), order="F")
            if solution.shape[:2] != (npe, nc):
                raise RuntimeError("HDG/LDG region output dimensions disagree")
            solution[:, :, ids] = output[-size:].reshape((npe, nc, owned), order="F")
            schemes[ids] = int(side == "H")
            assigned[ids] = True
    if not np.all(assigned):
        raise RuntimeError("HDG/LDG output omitted an original element")
    return (solution, schemes) if return_schemes else solution


def hdgldgsolve(pde, mesh, labels, nhdg=1):
    """Return ``(sol,pde,mesh)`` in original full-mesh element order.

    ``labels[e]`` is H=1 or L=0. The first nhdg MPI ranks own H elements;
    alternatively labels may be {"seed_boundaries": [1], "layers": 2}.
    Seeds are boundary-condition codes; layers expand through face neighbours.
    remaining ranks own L elements. Boundary expressions are scalar strings
    accepted by the shared CPU preprocessor (the exporttext2code convention).
    Pure native preprocessing is unchanged. Output uses q=-grad(u).
    Returned mesh["hdgldg"] contains H=1/L=0 in original element order,
    including when the selection was specified through boundary layers.

    Each run creates a fresh folder under pde['datapath'] (default cwd), with
    input/output and a normal hidden .exasim build. Set pde['hdgldgpath'] to
    choose a new explicit folder. Existing folders are never overwritten.
    """
    if pde.get("platform", "cpu") != "cpu" or pde.get("model", "ModelD").lower() != "modeld":
        raise ValueError("hdgldgsolve currently supports CPU ModelD cases")
    ranks = pde.get("mpiprocs", 1)
    if isinstance(ranks, bool) or int(ranks) != ranks or ranks < 2:
        raise ValueError("hdgldgsolve requires at least two MPI ranks")
    if isinstance(nhdg, bool) or int(nhdg) != nhdg or not 0 < nhdg < ranks:
        raise ValueError("nhdg must leave at least one rank for each scheme")
    ranks, nhdg = int(ranks), int(nhdg)
    ne = np.asarray(mesh["t"]).shape[1]
    if isinstance(labels, dict):
        seeds = np.asarray(labels.get("seed_boundaries", []), dtype=float).reshape(-1)
        layers = labels.get("layers", 0)
        if (not len(seeds) or not np.all(np.isfinite(seeds)) or
                not np.all((seeds >= 1) & (seeds == np.floor(seeds))) or
                not np.isscalar(layers) or not np.isfinite(layers) or
                layers < 1 or layers != int(layers)):
            raise ValueError("provide positive integer seed_boundaries and layers")
        selection = "selection=boundary_layers\nseed_boundaries=%s\nlayers=%d\n" % (
            ",".join(map(str, seeds.astype(int))), int(layers))
    else:
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != ne or not np.all(np.isin(labels, [0, 1])):
            raise ValueError("labels must contain one H=1/L=0 value per input element")
        if np.count_nonzero(labels) < nhdg or np.count_nonzero(labels == 0) < ranks - nhdg:
            raise ValueError("each scheme needs at least one owned element per rank")
        selection = "selection=labels\nlabels=%s\n" % ",".join(map(str, labels.astype(int)))
    preprocessor = config.install_prefix() / "bin" / "exasim_hdgldg_preprocess"
    if not preprocessor.is_file():
        raise FileNotFoundError("installed Exasim lacks " + str(preprocessor))
    parent = Path(pde.get("datapath", os.getcwd())).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    if pde.get("hdgldgpath"):
        root = Path(pde["hdgldgpath"]).resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(tempfile.mkdtemp(prefix="hdgldg-", dir=parent))
    pde["hdgldgpath"] = str(root)
    pde["builddir"] = str(root / ".exasim")
    pde["buildpath"] = pde["builddir"]
    pde["hybrid"] = 1  # Native gencode emits both LDG and HDG model kernels.
    pde["saveSolOpt"] = 1
    pde["saveSolFreq"] = 1
    pde["builtinmodelID"] = config.resolve_modelid(pde)
    _infer_dimensions(pde, mesh)
    pde["nch"] = pde["ncu"]
    pde["uinf"] = np.asarray(pde.get("externalparam", []))
    if pde.get("pgauss", 0) <= 0:
        pde["pgauss"] = 2 * pde["porder"]
    # Convert only the exported COPY: Python native t is zero-based, while
    # the existing shared binary grid reader consumes one-based connectivity.
    exported_mesh = copy.deepcopy(mesh)
    exported_mesh["t"] = np.asarray(mesh["t"], dtype=np.int64) + 1
    source = root / "source"
    Gencode.exporttext2code(pde, exported_mesh, source)
    # These explicit native metadata fields are not emitted by the older
    # general exporter. Add them only to this new generated configuration.
    configuration = source / "pdeapp.txt"
    emitted = {line.partition("=")[0].strip() for line in configuration.read_text().splitlines() if "=" in line}
    with configuration.open("a", encoding="utf-8") as stream:
        for name in ("nsca", "nvec", "nten", "nsurf", "nvqoi", "nmaterialstate", "nsurfq"):
            if pde.get(name, 0):
                raise ValueError("hdgldgsolve currently requires zero " + name)
            if name not in emitted:
                stream.write(name + " = 0;\n")
    manifest = root / "selection.txt"
    manifest.write_text("version=1\nworld_size=%d\nh_ranks=%s\nl_ranks=%s\n" %
                        (ranks, ",".join(map(str, range(nhdg))),
                         ",".join(map(str, range(nhdg, ranks)))) + selection, encoding="utf-8")
    datain, dataout = root / "input", root / "output"
    Preprocessing.hdgldgpreprocessing(source / "pdeapp.txt", manifest, datain, preprocessor)
    Gencode.gencode(pde)
    Gencode.cmakecompile(pde)
    build = Path(config.model_builddir(pde)) / "build"
    executable = build / "exasimapp"
    launcher = pde.get("mpirun", "mpirun")
    # Same launcher choice as native Gencode.runcode: prefer CMake's record.
    if (build / "mpiexec.txt").is_file():
        launcher = (build / "mpiexec.txt").read_text().strip() or launcher
    subprocess.run([launcher, "-np", str(ranks), str(executable),
                    "--hdgldg", str(datain), str(dataout)], check=True)
    solution, mesh["hdgldg"] = _collect_solution(datain, dataout, ne, return_schemes=True)
    pde["dataoutpath"] = str(dataout)
    print("HDG/LDG run files: " + str(root))
    return solution, pde, mesh
