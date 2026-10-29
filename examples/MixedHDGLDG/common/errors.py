"""Independent tensor-element RMS errors for native Exasim output.

The exact callback takes physical points [nq,nd] and returns [nq,nc], with
columns [u_0,...,u_(ncu-1),q_x0,...,q_x(ncu-1),q_y0,...]. Exasim uses
q=-grad(u). No solver shape, quadrature or mass arrays enter the integration.
This intentionally supports quad/hex tensor geometry only; it rejects rather
than guesses simplex, incomplete output, duplicate ownership or cut elements.
"""

from pathlib import Path
import json
import math

import numpy as np
from numpy.polynomial.legendre import leggauss


def sections(path):
    """Read the native double size-table format, with exact extent checks."""
    values = np.fromfile(path, dtype=np.float64)
    if not len(values) or not np.isfinite(values[0]) or values[0] != int(values[0]):
        raise ValueError("invalid native size table: " + str(path))
    count = int(values[0])
    if count < 1 or len(values) < count + 1:
        raise ValueError("truncated native size table: " + str(path))
    sizes = values[1:count + 1]
    if not np.all(np.isfinite(sizes)) or np.any(sizes < 0) or np.any(sizes != sizes.astype(int)):
        raise ValueError("invalid native section sizes: " + str(path))
    sizes = sizes.astype(int)
    if 1 + count + sum(sizes) != len(values):
        raise ValueError("native payload size mismatch: " + str(path))
    offsets = count + 1 + np.r_[0, np.cumsum(sizes)]
    return [values[offsets[k]:offsets[k + 1]] for k in range(count)]


def _integers(values, description):
    if not np.all(np.isfinite(values)) or np.any(values != values.astype(np.int64)):
        raise ValueError("invalid integer array: " + description)
    return values.astype(np.int64)


def _sidecar(path):
    tokens = path.read_text().split()
    if tokens[:2] != ["EXASIM_HDGLDG", "1"]:
        raise ValueError("unsupported region sidecar: " + str(path))
    scheme, world, local, selector, boundary = map(int, tokens[2:7])
    count = int(tokens[7])
    entries = np.array(list(map(int, tokens[8:8 + 3 * count])), dtype=np.int64).reshape(count, 3)
    if not np.array_equal(entries[:, 0], np.arange(count)):
        raise ValueError("noncanonical local element indices in " + str(path))
    return {"scheme": scheme, "world": world, "local": local,
            "ids": entries[:, 1], "owned": entries[:, 2] == world}


def load_partition(datain, fields, local_rank=0, streamed=True, global_ids=None):
    """Load one rank's OWNED fields and corresponding actual geometry.

    A one-rank native region uses mesh.bin/sol.bin; a parallel region uses
    mesh1.bin/sol1.bin for local rank zero, etc. A streamed solution has the
    native three-double [npe,nc,ownedElements] header; the last complete frame
    is selected. Raw restart checkpoints have no header (streamed=False).
    """
    datain, fields = Path(datain), Path(fields)
    suffix = str(local_rank + 1) if (datain / "mesh1.bin").exists() else ""
    mesh = sections(datain / ("mesh" + suffix + ".bin"))
    sol = sections(datain / ("sol" + suffix + ".bin"))
    master = sections(datain / "master.bin")
    dimensions = _integers(sol[0], "sol.ndims")
    ne, npe, nc, ncu, nd = (int(dimensions[k]) for k in (0, 3, 5, 6, 11))
    if nd not in (2, 3) or nc != ncu * (1 + nd):
        raise ValueError("error integration requires u and every q=-grad(u) component")
    if len(sol[1]) != npe * nd * ne or len(master[9]) != npe * nd:
        raise ValueError("geometry/master dimensions disagree")
    xdg = sol[1].reshape((npe, nd, ne), order="F")
    nodes = master[9].reshape((npe, nd), order="F")
    parts = _integers(mesh[9], "mesh.elempart")
    if len(parts) != ne:
        raise ValueError("native element permutation length disagrees with sol")
    points = _integers(mesh[10], "mesh.elempartpts")
    owned = int(sum(points[:2])) if suffix else ne
    if not 0 <= owned <= ne:
        raise ValueError("invalid native owned-element count")
    sidecar_path = datain / ("hdgldg" + suffix + ".txt")
    if sidecar_path.exists():
        metadata = _sidecar(sidecar_path)
        if len(metadata["ids"]) != ne or not np.array_equal(metadata["owned"], np.arange(ne) < owned):
            raise ValueError("sidecar ownership disagrees with native owned prefix")
        ids = metadata["ids"][:owned]
    else:
        # Native elempart uses zero-based full input indices. The caller may
        # provide the canonical mesh's persistent-ID vector for pure schemes.
        ids = parts[:owned] if global_ids is None else np.asarray(global_ids)[parts[:owned]]
    raw = np.fromfile(fields, dtype=np.float64)
    frame = npe * nc * owned
    if streamed:
        if len(raw) < 3 or not np.array_equal(raw[:3], [npe, nc, owned]):
            raise ValueError("streamed output must contain saveSolOpt=1 full-nc fields")
        raw = raw[3:]
        if not frame or len(raw) < frame or len(raw) % frame:
            raise ValueError("truncated/empty streamed solution frame")
        raw = raw[-frame:]
    elif len(raw) != frame:
        raise ValueError("raw checkpoint extent disagrees with owned full-nc fields")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(xdg)):
        raise ValueError("nonfinite solution or geometry")
    return {"xdg": xdg[:, :, :owned], "udg": raw.reshape((npe, nc, owned), order="F"),
            "nodes": nodes, "ids": np.asarray(ids, dtype=np.int64), "ncu": ncu,
            "input": str(datain), "fields": str(fields)}


def tensor_basis(nodes, points):
    """Lagrange values and first derivatives in the master's actual node order."""
    npe, nd = nodes.shape
    axes = [np.unique(nodes[:, d]) for d in range(nd)]
    if any(len(axis) != len(axes[0]) for axis in axes) or len(axes[0]) ** nd != npe:
        raise ValueError("only complete equal-degree tensor elements are supported")
    if len(axes[0]) < 2 or any(axis[0] != 0 or axis[-1] != 1 for axis in axes):
        raise ValueError("expected native tensor reference cell [0,1]^nd, p>=1")
    values, derivatives, indices = [], [], []
    for d, axis in enumerate(axes):
        degree = len(axis)
        v, dv = np.ones((len(points), degree)), np.zeros((len(points), degree))
        for j in range(degree):
            for k in range(degree):
                if k != j:
                    v[:, j] *= (points[:, d] - axis[k]) / (axis[j] - axis[k])
            for k in range(degree):
                if k == j:
                    continue
                term = np.full(len(points), 1 / (axis[j] - axis[k]))
                for m in range(degree):
                    if m != j and m != k:
                        term *= (points[:, d] - axis[m]) / (axis[j] - axis[m])
                dv[:, j] += term
        values.append(v); derivatives.append(dv)
        indices.append(np.searchsorted(axis, nodes[:, d]))
    tuples = np.stack(indices, axis=1)
    if len(np.unique(tuples, axis=0)) != npe:
        raise ValueError("duplicate reference tensor node")
    basis = np.ones((len(points), npe))
    for d in range(nd):
        basis *= values[d][:, indices[d]]
    gradient = np.ones((len(points), npe, nd))
    for d in range(nd):
        for k in range(nd):
            gradient[:, :, d] *= (derivatives[k] if k == d else values[k])[:, indices[k]]
    return basis, gradient


def integrate(partitions, exact, nquad, split=.5):
    """Integrate component squared errors on the global and fixed H/L regions.

    Every element must lie wholly on one side of x=split. This guarantees
    that both pure and mixed results use exactly the same physical regions.
    """
    result = {name: {"volume": 0., "elements": 0, "squared": None} for name in ("global", "H", "L")}
    seen = set()
    for partition in partitions:
        xdg, udg, nodes = (partition[k] for k in ("xdg", "udg", "nodes"))
        nd, nc = xdg.shape[1], udg.shape[1]
        z, w = leggauss(nquad); z, w = (z + 1) / 2, w / 2
        multi = np.array(np.meshgrid(*([np.arange(nquad)] * nd), indexing="ij")).reshape(nd, -1).T
        quadrature = z[multi]
        weight = np.prod(w[multi], axis=1)
        basis, gradient = tensor_basis(nodes, quadrature)
        for e, element_id in enumerate(partition["ids"]):
            if int(element_id) in seen:
                raise ValueError("duplicate owned persistent element ID")
            seen.add(int(element_id))
            coords = xdg[:, :, e]
            if coords[:, 0].max() <= split + 64 * np.finfo(float).eps:
                region = "H"
            elif coords[:, 0].min() >= split - 64 * np.finfo(float).eps:
                region = "L"
            else:
                raise ValueError("H/L split cuts an element; use a face-aligned fixed region")
            points = basis @ coords
            jacobian = np.einsum("qjd,ja->qad", gradient, coords)
            determinant = np.linalg.det(jacobian)
            if not np.all(np.isfinite(determinant)) or np.any(determinant <= 0):
                raise ValueError("nonpositive physical mapping Jacobian")
            weighted = weight * determinant
            reference = np.asarray(exact(points), dtype=float)
            if reference.shape != (len(points), nc) or not np.all(np.isfinite(reference)):
                raise ValueError("exact callback must return all finite u/q components")
            difference = basis @ udg[:, :, e] - reference
            squared = np.einsum("q,qc,qc->c", weighted, difference, difference)
            for name in ("global", region):
                entry = result[name]
                entry["volume"] += float(sum(weighted)); entry["elements"] += 1
                entry["squared"] = squared.copy() if entry["squared"] is None else entry["squared"] + squared
    for entry in result.values():
        entry["rms"] = None if not entry["elements"] else np.sqrt(entry["squared"] / entry["volume"]).tolist()
        entry["squared"] = None if entry["squared"] is None else entry["squared"].tolist()
    return result


def measure(partitions, exact, start=None, relative=1e-8, absolute=1e-14, max_increment=8, split=.5):
    """Record quadrature refinement, not just the solver's integration rule.

    Defaults compare p+3,p+5,... points/direction. The measurement-stability
    controls do not alter any solver tolerance or spatial acceptance target.
    The caller must inspect `quadrature_stable` before reporting rates.
    """
    p = round(partitions[0]["nodes"].shape[0] ** (1 / partitions[0]["nodes"].shape[1])) - 1
    first = p + 3 if start is None else start
    if first < p + 3:
        raise ValueError("independent error integration must start at least p+3")
    history, previous, stable = [], None, False
    for nquad in range(first, first + max_increment + 1, 2):
        value = integrate(partitions, exact, nquad, split)
        difference = 0.
        if previous is not None:
            stable = True
            for name in value:
                if value[name]["rms"] is None:
                    continue
                a, b = np.array(value[name]["rms"]), np.array(previous[name]["rms"])
                delta = abs(a - b); difference = max(difference, float(max(delta)))
                stable &= bool(np.all(delta <= absolute + relative * np.maximum(abs(a), abs(b))))
        history.append({"points_per_direction": nquad, "errors": value,
                        "max_error_change": None if previous is None else difference})
        if stable:
            break
        previous = value
    return {"p": p, "q_convention": "q=-grad(u), direction-major components",
            "quadrature_stable": stable, "measurement_relative": relative,
            "measurement_absolute": absolute, "quadrature": history, "errors": history[-1]["errors"]}


def rates(coarse, fine, h_coarse, h_fine, roundoff_floor=0.):
    """Component interval rates, suppressing a caller-justified roundoff floor.

    No universal dimensional roundoff scale is assumed here. The benchmark
    must supply a scale-derived floor for exact/roundoff components; zero is
    suitable only when every reported component is genuinely resolved.
    """
    if not h_coarse > h_fine > 0:
        raise ValueError("rates require actual decreasing positive h")
    return [None if a <= roundoff_floor or b <= roundoff_floor else math.log(a / b) / math.log(h_coarse / h_fine)
            for a, b in zip(coarse, fine)]


def self_test():
    """Independent tensor polynomial and perturbed-state RMS moment checks."""
    maximum = 0.
    for nd in (2, 3):
        for p in (1, 2, 3):
            points = np.array(np.meshgrid(*([np.linspace(0, 1, p + 1)] * nd), indexing="ij")).reshape(nd, -1).T
            # Reverse the master node order to ensure no lexicographic assumption.
            nodes = points[::-1].copy()
            def exact(x):
                u = 1 + np.sum(x ** p, axis=1)
                return np.column_stack([u] + [-p * x[:, d] ** (p - 1) for d in range(nd)])
            # Two cells, with an exact split and non-unit physical x scale.
            geometry = np.stack([nodes * np.array([.5] + [1.] * (nd - 1)) + np.array([shift] + [0.] * (nd - 1)) for shift in (0., .5)], axis=2)
            field = np.stack([exact(geometry[:, :, e]) for e in (0, 1)], axis=2)
            partition = {"xdg": geometry, "udg": field, "nodes": nodes, "ids": np.array([71, 95]), "ncu": 1}
            result = measure([partition], exact)
            maximum = max(maximum, max(result["errors"]["global"]["rms"]))
            assert result["quadrature_stable"] and maximum < 1e-13
            assert abs(result["errors"]["H"]["volume"] - .5) < 1e-14
            field[:, 0, :] += .125
            result = measure([partition], exact)
            assert abs(result["errors"]["global"]["rms"][0] - .125) < 1e-14
    print(json.dumps({"independent_tensor_self_test": "PASS", "polynomial_rms_max": maximum}))


if __name__ == "__main__":
    self_test()
