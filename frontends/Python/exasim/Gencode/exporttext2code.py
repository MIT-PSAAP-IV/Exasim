import os
import numpy as np

from .genpdemodel import genpdemodel


def exporttext2code(pde, mesh, dest=None):
    """Export a self-contained Text2Code application package.

    The package contains ``pdemodel.txt``, ``pdeapp.txt``, ``grid.bin`` and
    optional field binaries (``xdg.bin``, ``udg.bin``, ``vdg.bin``, ``wdg.bin``)
    when the corresponding mesh arrays are present.

    This is intentionally different from ``exportapp``: it exports the
    high-level Text2Code inputs, not generated kernels or a C++ driver.
    """
    if dest is None or dest == "":
        dest = pde.get("exporttext2code", pde.get("exportt2c", ""))
    if dest is None or dest == "":
        raise ValueError("exporttext2code: no destination (pass dest or set pde['exporttext2code']).")

    dest = os.path.abspath(os.fspath(dest))
    os.makedirs(dest, exist_ok=True)

    app = dict(pde)
    _infer_dimensions(app, mesh)

    print(f"Export Exasim Text2Code package to {dest} ...")
    genpdemodel(app, os.path.join(dest, "pdemodel.txt"))

    files = _write_binaries(mesh, dest)
    _write_pdeapp(app, mesh, files, os.path.join(dest, "pdeapp.txt"))
    _write_readme(dest)

    print(f"Exported Text2Code package: {dest}")
    return dest


def _write_binaries(mesh, dest):
    files = {"meshfile": "grid.bin"}
    p = _mesh_get(mesh, "p")
    t = _mesh_get(mesh, "t")
    if p is None or t is None:
        raise ValueError("exporttext2code: mesh must contain p and t arrays.")
    _writebin(os.path.join(dest, "grid.bin"), np.concatenate((_shape_as_double(p), _shape_as_double(t), _flat(p), _flat(t))))

    optional = [
        ("dgnodes", "xdgfile", "xdg.bin"),
        ("udg", "udgfile", "udg.bin"),
        ("vdg", "vdgfile", "vdg.bin"),
        ("wdg", "wdgfile", "wdg.bin"),
    ]
    for mesh_key, app_key, filename in optional:
        value = _mesh_get(mesh, mesh_key)
        if value is not None and not _empty(value):
            _writebin(os.path.join(dest, filename), np.concatenate((_shape_as_double(value), _flat(value))))
            files[app_key] = filename
    return files


def _infer_dimensions(app, mesh):
    p = _mesh_get(mesh, "p")
    if p is not None:
        nd = int(np.shape(p)[0])
        app["nd"] = nd
        app["ncx"] = nd
    else:
        nd = int(app.get("nd", 1))

    ncu = int(app.get("ncu", 1))
    model = str(app.get("model", app.get("pdemodel", "ModelD"))).lower()
    if model == "modelc":
        nc = ncu
    else:
        nc = ncu * (nd + 1)
    app["nc"] = int(app.get("nc", nc))
    if app["nc"] < nc:
        app["nc"] = nc
    app["ncq"] = max(app["nc"] - ncu, 0)

    vdg = _mesh_get(mesh, "vdg")
    if vdg is not None and not _empty(vdg):
        app["nco"] = int(np.shape(vdg)[1])
    wdg = _mesh_get(mesh, "wdg")
    if wdg is not None and not _empty(wdg):
        app["ncw"] = int(np.shape(wdg)[1])


def _write_pdeapp(pde, mesh, files, path):
    app = dict(pde)
    app["modelfile"] = "pdemodel.txt"
    app["meshfile"] = files["meshfile"]
    for key in ("xdgfile", "udgfile", "vdgfile", "wdgfile"):
        if key in files:
            app[key] = files[key]
        elif key in app:
            del app[key]

    app["discretization"] = "hdg" if int(app.get("hybrid", 0)) == 1 else "ldg"
    app["NewtonIter"] = app.get("NLiter", app.get("NewtonIter", 20))
    app["NewtonTol"] = app.get("NLtol", app.get("NewtonTol", 1e-6))
    app["GMRESiter"] = app.get("linearsolveriter", app.get("GMRESiter", 200))
    app["GMREStol"] = app.get("linearsolvertol", app.get("GMREStol", 1e-3))
    app["ncv"] = app.get("nco", app.get("ncv", 0))
    app["frontendgenerated"] = 0
    if not _empty(app.get("physicsparamsweep", [])):
        app["physicsparamcases"] = _normalize_sweep_cases(app["physicsparamsweep"], len(_as_1d(app.get("physicsparam", []))))

    boundaryconditions = _mesh_get(mesh, "boundarycondition")
    if boundaryconditions is not None:
        app["boundaryconditions"] = boundaryconditions
    app["boundaryexpressions"] = _string_list(_mesh_get(mesh, "boundaryexpr"), "boundaryexpr")

    curved = _mesh_get(mesh, "curvedboundary")
    app["curvedboundaries"] = curved if curved is not None else []
    curvedexpr = _mesh_get(mesh, "curvedboundaryexprs")
    if curvedexpr is None:
        curvedexpr = _mesh_get(mesh, "curvedboundaryexpr")
    if curvedexpr is None:
        curvedexpr = ["" for _ in _as_1d(app["boundaryconditions"])]
    app["curvedboundaryexprs"] = _string_list(curvedexpr, "curvedboundaryexpr")

    _add_periodic(app, mesh)
    interfaceconditions = _mesh_get(mesh, "interfacecondition")
    app["interfaceconditions"] = interfaceconditions if interfaceconditions is not None else []

    keys = [
        "model", "modelfile", "meshfile", "xdgfile", "udgfile", "vdgfile", "wdgfile",
        "discretization", "platform", "mpiprocs", "debugmode", "runmode", "modelnumber",
        "builtinmodelID", "frontendgenerated",
        "nodetype", "ncu", "ncv", "ncw", "neb", "nfb", "linearproblem", "subproblem",
        "saveParaview", "physicsparamwarmstart", "tdep", "wave", "porder", "pgauss",
        "temporalscheme", "torder", "nstage", "convStabMethod", "diffStabMethod",
        "rotatingFrame", "viscosityModel", "SGSmodel", "ALE", "AV", "AVsmoothingIter",
        "frozenAVflag", "nonlinearsolver", "linearsolver", "NewtonIter", "NewtonTol",
        "GMRESiter", "GMRESrestart", "GMREStol", "GMRESortho", "ppdegree", "RBdim",
        "matvecorder", "matvectol", "precMatrixType", "preconditioner", "time",
        "NLparam", "tau", "dt", "dae_alpha", "dae_beta", "dae_gamma", "dae_epsilon",
        "dae_steps", "dae_dt", "physicsparam", "physicsparamcases", "externalparam",
        "vindx", "avparam1", "avparam2", "stgib", "stgdata", "stgparam",
        "boundaryconditions", "boundaryexpressions", "curvedboundaries",
        "curvedboundaryexprs", "periodicboundaries1", "periodicexprs1",
        "periodicboundaries2", "periodicexprs2", "interfaceconditions",
        "interfacefluxmap", "wmModelIDs", "wmBoundaries", "wmDistances",
        "saveSolFreq", "saveSolOpt", "timestepOffset", "saveSolBouFreq", "ibs",
        "compudgavg", "extFhat", "extUhat", "extStab", "saveResNorm",
    ]

    with open(path, "w", encoding="utf-8") as f:
        for key in keys:
            if key in app and app[key] is not None and not _empty(app[key]):
                f.write(f"{key} = {_format_value(app[key], key)};\n")


def _add_periodic(app, mesh):
    periodic = _mesh_get(mesh, "periodicboundary")
    if periodic is None or len(periodic) == 0:
        app["periodicboundaries1"] = []
        app["periodicboundaries2"] = []
        app["periodicexprs1"] = []
        app["periodicexprs2"] = []
        return
    b1, b2, e1, e2 = [], [], [], []
    for row in periodic:
        b1.append(row[0])
        e1.extend(_periodic_expr(row[1]))
        b2.append(row[2])
        e2.extend(_periodic_expr(row[3]))
    app["periodicboundaries1"] = b1
    app["periodicboundaries2"] = b2
    app["periodicexprs1"] = e1
    app["periodicexprs2"] = e2


def _periodic_expr(expr):
    values = _string_list([expr], "periodic expression")
    if values[0] == "xy":
        return ["x", "y"]
    if values[0] == "xz":
        return ["x", "z"]
    if values[0] == "yz":
        return ["y", "z"]
    return values


def _mesh_get(mesh, key):
    if isinstance(mesh, dict):
        return mesh.get(key)
    return getattr(mesh, key, None)


def _writebin(filename, data):
    np.asarray(data, dtype=np.float64).flatten(order="F").tofile(filename)


def _shape_as_double(a):
    return np.asarray(np.shape(a), dtype=np.float64).flatten(order="F")


def _flat(a):
    return np.asarray(a, dtype=np.float64).flatten(order="F")


def _as_1d(value):
    if value is None:
        return []
    return np.asarray(value).flatten(order="F")


def _normalize_sweep_cases(spec, nparam):
    if isinstance(spec, dict):
        if "samples" in spec:
            return _normalize_sweep_cases(spec["samples"], nparam)
        if "values" in spec:
            return _normalize_sweep_cases(spec["values"], nparam)
        if "grid" in spec:
            grid = spec["grid"]
            if len(grid) != nparam:
                raise ValueError("physicsparamsweep['grid'] must have one value vector per physics parameter.")
            meshes = np.meshgrid(*[np.asarray(v).reshape(-1) for v in grid], indexing="ij")
            return np.stack([m.reshape(-1) for m in meshes], axis=1)
        raise ValueError("physicsparamsweep dict must contain samples, values, or grid.")
    arr = np.asarray(spec, dtype=np.float64)
    if arr.ndim == 1:
        if nparam == 1:
            arr = arr.reshape((-1, 1))
        else:
            arr = arr.reshape((1, -1))
    if arr.shape[1] != nparam:
        raise ValueError(f"Each physicsparamsweep row must contain {nparam} physics parameters.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("physicsparamsweep cases must contain finite numeric values.")
    return arr


def _empty(value):
    if isinstance(value, str):
        return value == ""
    try:
        return len(value) == 0
    except TypeError:
        return False


def _string_list(value, label):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    values = []
    for item in list(np.asarray(value, dtype=object).flatten(order="F")):
        if not isinstance(item, str):
            raise ValueError(f"exporttext2code: {label} entries must be strings for Text2Code export.")
        values.append(item)
    return values


def _format_value(value, key=None):
    if isinstance(value, str):
        return f"\"{value}\""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _format_float(float(value))
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    arr = np.asarray(value)
    if key == "physicsparamcases" and arr.ndim == 2:
        rows = []
        for row in arr:
            rows.append(" ".join(_format_float(float(v)) for v in row))
        return "[" + "; ".join(rows) + "]"
    values = list(arr.flatten(order="F"))
    if len(values) == 0:
        return "[]"
    if all(isinstance(v, str) for v in values):
        return "[" + ", ".join(f"\"{v}\"" for v in values) + "]"
    return "[" + ", ".join(_format_float(float(v)) for v in values) + "]"


def _format_float(value):
    return f"{value:.17g}"


def _write_readme(dest):
    text = """# Exasim Text2Code Export

This directory contains high-level Text2Code inputs exported from an Exasim frontend.

Generated files:

- `pdemodel.txt`: PDE model definition consumed by Text2Code.
- `pdeapp.txt`: application, mesh, solver, output, and runtime configuration.
- `grid.bin`: mesh coordinates and connectivity.
- `xdg.bin`, `udg.bin`, `vdg.bin`, `wdg.bin`: optional field data written only when present.

Regenerate the application with:

```sh
/path/to/exasim-prefix/bin/text2code pdeapp.txt
```

The `vdg.bin` file stores external variables. In backend data structures these are also called `odg`.
"""
    with open(os.path.join(dest, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)
