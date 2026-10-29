#!/usr/bin/env python3
"""Measure and compare completed example runs written by run.py.

measure ROOT [--quadrature-start K] : errors, solver history, balances -> ROOT/results.json
parity  ROOT_A ROOT_B ...           : same case/scheme across frontends
diagnostics are read from native output files only; no solver file is changed.
"""
import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import errors  # noqa: E402
import cases  # noqa: E402

EPS = np.finfo(float).eps


def regions(root):
    """Native partitions of a run: datain, output prefix, local rank, scheme."""
    run = json.loads((root / "run.json").read_text())
    found = []
    if run["scheme"] == "mixed":
        for group, scheme in (("H", "hdg"), ("L", "ldg")):
            datain = root / "prepared" / group
            count = len(list(datain.glob("mesh[0-9]*.bin"))) or 1
            for local in range(count):
                found.append({"group": group, "scheme": scheme, "datain": datain, "local": local,
                              "prefix": root / "output" / group / "out"})
    else:
        for local in range(run["ranks"]):
            found.append({"group": None, "scheme": run["scheme"], "datain": root / "datain", "local": local,
                          "prefix": root / "output" / "out" / "out"})
    return run, found


def load_fields(region):
    return errors.load_partition(region["datain"], "%sudg_np%d.bin" % (region["prefix"], region["local"]),
                                 region["local"])


def history(log):
    """Ordered nonlinear/linear records and warnings printed by the solvers."""
    text = log.read_text(errors="replace")
    config = (log.parent / "pdeapp.txt").read_text()
    tolerance = float(re.search(r"(?m)^\s*GMREStol\s*=\s*([^;\s]+)", config).group(1))
    gmres = [int(v) for v in re.findall(r"GMRES(?:\(\d+\))? finished after (\d+) iterations", text)]
    mixed = re.findall(r"HDGLDG Newton (\d+) GMRES (\d+) relative (\S+) alpha (\S+) residual (\S+) accepted (\d)", text)
    residuals = [float(v) for v in re.findall(r"Residual Norm: (\S+)", text)]
    updated = [float(v) for v in re.findall(r"Updated Norm: (\S+)", text)]
    alphas = [float(v) for v in re.findall(r"Alpha: (\S+),", text)]
    warnings = [line.strip() for line in text.splitlines()
                if re.search(r"warn|not converge|maximum number|limit|nan|inf\b|error", line, re.I)]
    if mixed:
        gmres = [int(m[1]) for m in mixed]
        alphas = [float(m[3]) for m in mixed]
        final = float(mixed[-1][4])
        updates = len(mixed)
    else:
        final = (updated or residuals or [math.nan])[-1]
        if residuals and updated:
            final = residuals[-1] if text.rfind("Residual Norm") > text.rfind("Updated Norm") else updated[-1]
        # One Newton update per linear solve; native LDG may print the same
        # iteration label twice, so labels are not counted.
        updates = len(gmres)
    relative = [float(m[2]) for m in mixed] or [float(v) for v in re.findall(r"relative error (\S+) \(effective", text)]
    return {"gmres": gmres, "gmres_total": sum(gmres), "linear_limit_exits": sum(r > tolerance for r in relative), "newton_updates": updates, "alphas": alphas,
            "final_residual": final, "linear_relative": [float(m[2]) for m in mixed] or
            [float(v) for v in re.findall(r"relative error (\S+) \(effective", text)],
            # A solve is a linear-limit exit when its reported relative
            # residual exceeds GMREStol (native warnings repeat per group).
            "linear_tolerance": tolerance,
            "printed_residuals": residuals, "printed_updated": updated, "warnings": warnings}


def read_block(path):
    raw = np.fromfile(path, dtype=np.float64)
    if len(raw) < 3:
        raise ValueError("truncated native boundary file " + str(path))
    npf, faces, comps = (int(v) for v in raw[:3])
    if len(raw) != 3 + npf*faces*comps:
        raise ValueError("native boundary extent mismatch " + str(path))
    return npf, faces, comps, raw[3:]


def owned_faces(region, fields=None):
    """Boundary records whose face lies on an element owned by this rank.

    Native LDG output also writes records for faces of first-ghost elements;
    those are not this rank's residual rows and are skipped (counted).
    """
    fields = load_fields(region) if fields is None else fields
    low, high = fields["xdg"].min(axis=0).T, fields["xdg"].max(axis=0).T
    tol = 1e-12*max(1., float(abs(fields["xdg"]).max()))
    owned, ghosts = [], 0
    for face in boundary_faces(region):
        a, b = face["xdg"].min(axis=0), face["xdg"].max(axis=0)
        if np.any(np.all((low - tol <= a) & (b <= high + tol), axis=1)):
            owned.append(face)
        else:
            ghosts += 1
    return owned, ghosts


def boundary_faces(region):
    """Per-face native boundary records with their boundary code, in file order."""
    prefix, local = region["prefix"], region["local"]
    name = lambda kind: Path("%sbou%s_np%d.bin" % (prefix, kind, local))
    info = np.fromfile(name("info"), dtype=np.float64)
    blocks = info[3:].reshape(-1, 2).astype(int) if len(info) > 3 else np.zeros((0, 2), int)
    data = {kind: read_block(name(kind)) for kind in ("xdg", "ndg", "udg", "uhat")}
    npf = data["xdg"][0]
    faces, offsets = [], {kind: 0 for kind in data}
    for code, count in blocks:
        arrays = {}
        for kind, (_, _, comps, raw) in data.items():
            size = npf*count*comps
            arrays[kind] = raw[offsets[kind]:offsets[kind] + size].reshape((npf, count, comps), order="F")
            offsets[kind] += size
        for f in range(count):
            faces.append({"code": int(code), **{k: v[:, f, :] for k, v in arrays.items()}})
    if any(offsets[k] != len(data[k][3]) for k in data):
        raise ValueError("boundary info blocks do not cover native records")
    return faces


def face_rule(x, points):
    """Lagrange interpolation matrix and weights on an affine axis-aligned face."""
    span = x.max(axis=0) - x.min(axis=0)
    scale = max(span.max(), 1.0)
    axes = [d for d in range(x.shape[1]) if span[d] > 1e3*EPS*scale]
    if len(axes) != x.shape[1] - 1:
        raise ValueError("balance diagnostics require axis-aligned straight faces")
    s = (x[:, axes] - x[:, axes].min(axis=0))/span[axes]
    nodes = [np.unique(np.round(s[:, k], 13)) for k in range(len(axes))]
    z, w = leggauss(points); z, w = (z + 1)/2, w/2
    grid = np.array(np.meshgrid(*([z]*len(axes)), indexing="ij")).reshape(len(axes), -1).T
    weight = np.prod(np.array(np.meshgrid(*([w]*len(axes)), indexing="ij")).reshape(len(axes), -1), axis=0)
    basis = np.ones((len(grid), len(x)))
    for k, axis_nodes in enumerate(nodes):
        if len(axis_nodes)**len(axes) != len(x):
            raise ValueError("face nodes are not a complete tensor set")
        for j in range(len(x)):
            for m in axis_nodes:
                if abs(m - s[j, k]) > 1e-12:
                    basis[:, j] *= (grid[:, k] - m)/(s[j, k] - m)
    return basis, weight*np.prod(span[axes])


def face_key(x):
    return tuple(np.round(np.sort(x, axis=0).ravel(), 11))


def balance(case, run, found, points=None):
    """Raw source/boundary/interface integrals in native residual convention."""
    model = cases.Case(case)
    ncu, p = model.ncu, run["porder"]
    rule = p + 1 if points is None else points           # solver pgauss=2p rule
    totals = {"source": np.zeros(ncu), "boundary": np.zeros(ncu), "HL_H": np.zeros(ncu), "HL_L": np.zeros(ncu)}
    magnitude = {key: np.zeros(ncu) for key in totals}
    hl, records, duplicates = {}, {}, 0
    for region in found:
        fields = load_fields(region)
        z, w = leggauss(rule); z, w = (z + 1)/2, w/2
        nd = fields["xdg"].shape[1]
        grid = np.array(np.meshgrid(*([z]*nd), indexing="ij")).reshape(nd, -1).T
        weight = np.prod(np.array(np.meshgrid(*([w]*nd), indexing="ij")).reshape(nd, -1), axis=0)
        basis, gradient = errors.tensor_basis(fields["nodes"], grid)
        for e in range(fields["xdg"].shape[2]):
            coords = fields["xdg"][:, :, e]
            jac = np.linalg.det(np.einsum("qjd,ja->qad", gradient, coords))
            terms = (weight*jac)[:, None]*model.source(basis @ coords)
            totals["source"] += terms.sum(axis=0); magnitude["source"] += abs(terms).sum(axis=0)
        interface = None
        sidecar = Path(region["datain"]) / ("hdgldg%s.txt" % ("" if not (Path(region["datain"]) / "mesh1.bin").exists()
                                                            else region["local"] + 1))
        if sidecar.exists():
            interface = int(sidecar.read_text().split()[6])
        faces, ghosts = owned_faces(region, fields)
        duplicates += ghosts
        for face in faces:
            key = face_key(face["xdg"])
            side = "HL" if face["code"] == interface else "bdry"
            tag = (region["group"], side, key)
            if tag in records:
                raise ValueError("a boundary face is owned twice")
            records[tag] = face
            B, weight_f = face_rule(face["xdg"], rule)
            xg, ng = B @ face["xdg"], B @ face["ndg"]
            ug, qg, uh = B @ face["udg"][:, :ncu], B @ face["udg"][:, ncu:], B @ face["uhat"]
            if side == "HL" or region["scheme"] == "hdg":
                fhat = model.hdg_flux(uh, ug, qg, ng)
            else:
                fhat = model.ldg_boundary_flux(xg, ug, qg, ng)
            terms = weight_f[:, None]*fhat
            target = "boundary" if side == "bdry" else "HL_" + region["group"]
            totals[target] += terms.sum(axis=0); magnitude[target] += abs(terms).sum(axis=0)
            if side == "HL":
                hl.setdefault(key, {})[region["group"]] = terms.sum(axis=0)
    pairs = [v for v in hl.values() if len(v) == 2]
    if len(pairs) != len(hl):
        raise ValueError("unpaired HL face records")
    face_imbalance = max((float(np.max(abs(v["H"] + v["L"]))) for v in pairs), default=0.)
    domain = totals["boundary"] + totals["HL_H"] + totals["HL_L"] - totals["source"]
    rounding = 4*EPS*sum(magnitude.values())*max(1, len(records))
    return {"rule_points_per_direction": rule, "components": ncu,
            "source_integral": totals["source"].tolist(), "physical_boundary_flux": totals["boundary"].tolist(),
            "HL_H_flux": totals["HL_H"].tolist(), "HL_L_flux": totals["HL_L"].tolist(),
            "interface_imbalance": (totals["HL_H"] + totals["HL_L"]).tolist(),
            "max_face_interface_imbalance": face_imbalance, "HL_faces": len(pairs),
            "domain_imbalance": domain.tolist(), "summation_rounding_bound": rounding.tolist(),
            "unique_boundary_records": len(records), "ignored_ghost_records": duplicates}


def degrees_of_freedom(found):
    total = 0
    for region in found:
        f = load_fields(region)
        total += f["udg"].shape[0]*f["udg"].shape[2]
    return total


def couette_walls(run, found):
    """Signed Couette wall diagnostics from u,q at the wall face nodes.

    n is the fluid's outward normal. tau_xy = (u_y+v_x)/Re is the fluid stress,
    work_on_fluid = int u.(tau.n) (pressure does no work: u.n = 0), and
    heat_into_fluid = int k dT/dn with k = 1/((gamma-1) M^2 Re Pr). These are
    physical diagnostics. numerical_energy_flux_out integrates the stabilized
    native boundary flux used by the residual; the two are not the same.
    """
    model = cases.Case("Couette")
    gam, Re, Pr, M = cases.PARAMETERS["Couette"][:4]
    k = 1/((gam - 1)*M*M*Re*Pr)
    result, seen = {}, set()
    for region in found:
        for face in owned_faces(region)[0]:
            x = face["xdg"]
            wall = ("lower" if np.all(abs(x[:, 1]) < 1e-12) else
                    "upper" if np.all(abs(x[:, 1] - 1) < 1e-12) else None)
            if wall is None or (wall, face_key(x)) in seen:
                continue
            seen.add((wall, face_key(x)))
            B, weight = face_rule(x, run["porder"] + 3)
            u, q, n, uh = B @ face["udg"][:, :4], B @ face["udg"][:, 4:], B @ face["ndg"], B @ face["uhat"]
            g = -q                                      # physical gradient
            r, vel = u[:, 0], u[:, 1:3]/u[:, :1]
            dvel = [(g[:, [1 + j, 5 + j]] - g[:, [0, 4]]*vel[:, [j]])/r[:, None] for j in (0, 1)]
            tau_xy = (dvel[0][:, 1] + dvel[1][:, 0])/Re
            p = (gam - 1)*(u[:, 3] - .5*r*(vel**2).sum(axis=1))
            dp = (gam - 1)*(g[:, [3, 7]] - .5*g[:, [0, 4]]*(vel**2).sum(axis=1)[:, None]
                            - r[:, None]*(vel[:, [0]]*dvel[0] + vel[:, [1]]*dvel[1]))
            dT = gam*M*M*(dp*r[:, None] - p[:, None]*g[:, [0, 4]])/(r*r)[:, None]
            entry = result.setdefault(wall, {"length": 0., "mean_tau_xy": 0., "work_on_fluid": 0.,
                                             "heat_into_fluid": 0., "numerical_energy_flux_out": 0.})
            entry["length"] += float(weight.sum())
            entry["mean_tau_xy"] += float(weight @ tau_xy)
            entry["work_on_fluid"] += float(weight @ (vel[:, 0]*tau_xy*n[:, 1] + vel[:, 1]*tau_xy*n[:, 0]))
            entry["heat_into_fluid"] += float(weight @ (k*(dT*n).sum(axis=1)))
            fhat = (model.hdg_flux(uh, u, q, n) if region["scheme"] == "hdg"
                    else model.ldg_boundary_flux(B @ x, u, q, n))
            entry["numerical_energy_flux_out"] += float(weight @ fhat[:, 3])
    for entry in result.values():
        entry["mean_tau_xy"] /= entry["length"]
    U, T0, T1 = cases.PARAMETERS["Couette"][4:7]
    a = (gam - 1)*M*M*Pr*U*U/2
    result["exact"] = {"tau_xy": U/Re, "upper_work_on_fluid": U*U/Re, "lower_heat_into_fluid": -k*(T1 - T0 + a),
                       "upper_heat_into_fluid": k*(T1 - T0 - a)}
    result["physical_energy_balance"] = sum(result[w]["work_on_fluid"] + result[w]["heat_into_fluid"]
                                            for w in ("lower", "upper") if w in result)
    return result


def curved_face_rule(x, nodes, points):
    """Interpolation and |dx/ds| weights on a 2D curved face (1D master nodes)."""
    z, w = leggauss(points); z, w = (z + 1)/2, w/2
    if len(nodes) != len(x):
        raise ValueError("face node count disagrees with master")
    basis, derivative = np.ones((points, len(x))), np.zeros((points, len(x)))
    for j in range(len(x)):
        for m in range(len(x)):
            if m != j:
                basis[:, j] *= (z - nodes[m])/(nodes[j] - nodes[m])
        for k in range(len(x)):
            if k == j:
                continue
            term = np.full(points, 1/(nodes[j] - nodes[k]))
            for m in range(len(x)):
                if m != j and m != k:
                    term *= (z - nodes[m])/(nodes[j] - nodes[m])
            derivative[:, j] += term
    return basis, w*np.linalg.norm(derivative @ x, axis=1)


def naca(run, found):
    """Admissibility and wall forces for the steady NACA case.

    The force on the airfoil is the integral of the fluid momentum flux
    F(u,q).n over the wall (n points out of the fluid, into the airfoil);
    at the no-slip wall this is p n - tau.n. Coefficients use the donor's
    rho_inf = U_inf = chord = 1. The numerical force integrates the native
    stabilized boundary flux instead; both are reported.
    """
    mu = cases.PARAMETERS["NACA2D"]
    gam = mu[0]
    out = {"min_density": math.inf, "min_pressure": math.inf, "finite": True,
           "wall_length": 0., "force_physical": [0., 0.], "force_numerical": [0., 0.]}
    model = cases.Case.__new__(cases.Case)
    model.name = "NACA2D"
    seen = set()
    for region in found:
        fields = load_fields(region)
        u = fields["udg"][:, :4, :]
        out["finite"] &= bool(np.all(np.isfinite(fields["udg"])))
        pressure = (gam - 1)*(u[:, 3] - .5*(u[:, 1]**2 + u[:, 2]**2)/u[:, 0])
        out["min_density"] = min(out["min_density"], float(u[:, 0].min()))
        out["min_pressure"] = min(out["min_pressure"], float(pressure.min()))
        nodes = np.unique(np.round(fields["nodes"][:, 0], 14))
        for face in owned_faces(region, fields)[0]:
            if face["code"] != 1 or face_key(face["xdg"]) in seen:
                continue
            seen.add(face_key(face["xdg"]))
            B, weight = curved_face_rule(face["xdg"], nodes, run["porder"] + 3)
            ug, qg, ng, uh = B @ face["udg"][:, :4], B @ face["udg"][:, 4:], B @ face["ndg"], B @ face["uhat"]
            ng = ng/np.linalg.norm(ng, axis=1)[:, None]
            F = cases.ns_flux(ug, qg, mu)
            physical = np.einsum("ncd,nd->nc", F, ng)
            if region["scheme"] == "hdg":
                numerical = np.einsum("ncd,nd->nc", cases.ns_flux(uh, qg, mu), ng) + cases.TAU["NACA2D"]*(ug - uh)
            else:
                numerical = physical + cases.TAU["NACA2D"]*(ug - uh)   # donor wall Fbou
            out["wall_length"] += float(weight.sum())
            for k in (0, 1):
                out["force_physical"][k] += float(weight @ physical[:, 1 + k])
                out["force_numerical"][k] += float(weight @ numerical[:, 1 + k])
    for kind in ("physical", "numerical"):
        drag, lift = out["force_" + kind]
        out["CD_" + kind], out["CL_" + kind] = 2*drag, 2*lift
    return out


def measure(root, start=None):
    run, found = regions(root)
    model = None if run["case"] == "NACA2D" else cases.Case(run["case"])
    partitions = [load_fields(r) for r in found]
    out = {"run": run, "history": history(root / "solver.log")}
    if run["case"] == "NACA2D":
        out["naca"] = naca(run, found)
        out["mesh"] = {"elements": sum(p["udg"].shape[2] for p in partitions)}
        (root / "results.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
        return out
    out["errors"] = errors.measure(partitions, model.exact, start=start)
    out["mesh"] = {"elements": sum(p["udg"].shape[2] for p in partitions), "h": 1/run["N"],
                   "dofs_per_component": degrees_of_freedom(found)}
    if run["case"] != "NACA2D":
        out["balance_solver_rule"] = balance(run["case"], run, found)
        out["balance_high_rule"] = balance(run["case"], run, found, run["porder"] + 6)
        n = out["mesh"]["dofs_per_component"]
        out["balance_residual_bound"] = math.sqrt(model.ncu*n)*out["history"]["final_residual"]
    if run["case"] == "Couette":
        out["couette_walls"] = couette_walls(run, found)
    (root / "results.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def parity(roots):
    """Same-scheme cross-route comparison by persistent IDs and coordinates."""
    base = None
    rows = []
    for root in roots:
        run, found = regions(root)
        fields = {}
        for region in found:
            f = load_fields(region)
            for e, i in enumerate(f["ids"]):
                fields[int(i)] = (f["xdg"][:, :, e], f["udg"][:, :, e])
        record = {"root": str(root), "frontend": run["frontend"], "history": history(root / "solver.log"),
                  "fields": fields, "run": run}
        if base is None:
            base = record
        else:
            if set(fields) != set(base["fields"]):
                raise ValueError("persistent element sets differ: " + str(root))
            dx = max(float(abs(fields[i][0] - base["fields"][i][0]).max()) for i in fields)
            du = max(float(abs(fields[i][1] - base["fields"][i][1]).max()) for i in fields)
            scale = max(float(abs(base["fields"][i][1]).max()) for i in fields)
            x0, u0 = fields[next(iter(fields))]
            ncu = u0.shape[1]//(1 + x0.shape[1])            # nc = ncu*(1+nd)
            dstate = max(float(abs(fields[i][1][:, :ncu] - base["fields"][i][1][:, :ncu]).max()) for i in fields)
            rows.append({"reference": base["frontend"], "frontend": run["frontend"], "scheme": run["scheme"],
                         "max_geometry_difference": dx, "max_field_difference": du, "field_scale": scale,
                         "max_state_difference": dstate,
                         "gmres_reference": base["history"]["gmres"], "gmres": record["history"]["gmres"],
                         "final_residual_reference": base["history"]["final_residual"],
                         "final_residual": record["history"]["final_residual"]})
    return rows


def input_files(root):
    """Native solver inputs written by the shared preprocessing for a run."""
    base = root / ("prepared" if (root / "prepared").exists() else "datain")
    return {str(f.relative_to(base)): f.read_bytes() for f in sorted(base.rglob("*")) if f.is_file()}


def parity_matrix(campaign):
    """Four-route parity on the campaign rows of one small matched case.

    Criterion fixed before inspecting results: identical shared solver inputs
    (bytes), identical Newton update counts, and route field differences at
    most 1e-3 of the same run's global discretization error (NACA: 1e-3 of
    the reference's field-scale-relative final residual is not meaningful, so
    its mapped differences and force differences are reported only).
    """
    rows = json.loads((campaign / "campaign.json").read_text())
    table = []
    groups = {}
    for row in rows:
        groups.setdefault((row["case"], row["scheme"]), []).append(row)
    for (case, scheme), members in sorted(groups.items()):
        ok = [m for m in members if m["run_status"] == 0 and m.get("measure_status") == 0]
        failed = [{"frontend": m["frontend"], "status": m["run_status"]} for m in members if m not in ok]
        ok.sort(key=lambda m: ("python", "matlab", "julia", "text2code").index(m["frontend"]))
        entry = {"case": case, "scheme": scheme, "failed_routes": failed, "routes": []}
        if ok:
            reference = Path(ok[0]["root"])
            ref_inputs = input_files(reference)
            ref_result = json.loads((reference / "results.json").read_text())
            rows_parity = parity([Path(m["root"]) for m in ok])
            for m, comparison in zip(ok[1:], rows_parity):
                root = Path(m["root"])
                result = json.loads((root / "results.json").read_text())
                inputs = input_files(root)
                differing = sorted(k for k in set(inputs) | set(ref_inputs) if inputs.get(k) != ref_inputs.get(k))
                record = dict(comparison, input_files_differing=differing,
                              newton_updates=(ref_result["history"]["newton_updates"], result["history"]["newton_updates"]))
                if "errors" in ref_result:
                    error = max(ref_result["errors"]["errors"]["global"]["rms"])
                    record["reference_max_global_error"] = error
                    record["difference_to_error_ratio"] = comparison["max_field_difference"]/error
                    record["pass"] = (not differing and record["newton_updates"][0] == record["newton_updates"][1]
                                      and record["difference_to_error_ratio"] <= 1e-3)
                else:
                    record["forces"] = (ref_result["naca"]["force_physical"], result["naca"]["force_physical"])
                    record["pass"] = None
                entry["routes"].append(record)
        table.append(entry)
    (campaign / "parity.json").write_text(json.dumps(table, indent=1) + "\n")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure"); m.add_argument("root", type=Path); m.add_argument("--quadrature-start", type=int)
    q = sub.add_parser("parity"); q.add_argument("roots", type=Path, nargs="+"); q.add_argument("--output", type=Path)
    t = sub.add_parser("parity-matrix"); t.add_argument("campaign", type=Path)
    args = parser.parse_args()
    if args.command == "measure":
        result = measure(args.root.resolve(), args.quadrature_start)
        summary = {"history": {k: result["history"][k] for k in ("gmres", "final_residual")}}
        if "errors" in result:
            summary.update(errors_global=result["errors"]["errors"]["global"]["rms"],
                           quadrature_stable=result["errors"]["quadrature_stable"])
        else:
            summary.update(result["naca"])
        print(json.dumps(summary))
    elif args.command == "parity-matrix":
        for entry in parity_matrix(args.campaign.resolve()):
            print(entry["case"], entry["scheme"], "failed:", entry["failed_routes"],
                  [(r["frontend"], r["pass"], r["max_field_difference"], r["input_files_differing"][:3])
                   for r in entry["routes"]])
    else:
        rows = parity([r.resolve() for r in args.roots])
        text = json.dumps(rows, indent=1)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
