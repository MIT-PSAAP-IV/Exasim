#!/usr/bin/env python3
"""Report matched Phase 5 frontend differences without a numerical gate.

Structure is compared exactly. Scalar solution entries are put in a common
physical order using (domain, element centroid, DG-node coordinate). Coordinate
matching is structural: exact values are preferred; otherwise corresponding
coordinates must be unique and differ by no more than 64*eps times their
physical scale. That rule is not applied to solution values.
"""
import argparse
import json
import pathlib

import numpy as np


EXACT_FIELDS = (
    "case", "dimension", "families", "N", "porder", "pgauss",
    "domain_ranks", "interaction_table", "object_counts", "element_counts",
    "interface_counts", "per_rank_interface_counts", "newton_iterations",
    "gmres_iterations", "solver_controls",
)

CANONICAL_TABLE = [
    1, 4, 3,
    11, 0, 1, 1, 2, 1, 1, 0, 4, 2, 1, 0,
    11, 1, 2, 1, 2, 2, 1, 0, 4, 1, 1, 0,
    11, 2, 3, 1, 2, 1, 1, 0, 4, 2, 1, 0,
]


def load_result(path):
    manifest = json.loads((path / "phase5_manifest.json").read_text())
    measured_fields = ("per_domain_l2", "combined_l2", "max_nodal_error",
                       "final_residual")
    missing = [key for key in EXACT_FIELDS + ("newton_history",) + measured_fields
               if key not in manifest]
    if missing:
        raise SystemExit(f"FAIL: {path} manifest misses {missing}")
    if manifest["interaction_table"] != CANONICAL_TABLE:
        raise SystemExit(f"FAIL: {path} does not contain the canonical interaction table")
    if manifest["object_counts"] != [1, 2, 2, 1]:
        raise SystemExit(f"FAIL: {path} object_counts differ from [1,2,2,1]")
    domain_fields = ("families", "domain_ranks", "object_counts", "element_counts",
                     "interface_counts", "per_rank_interface_counts", "per_domain_l2",
                     "final_residual", "newton_iterations", "newton_history")
    if any(len(manifest[field]) != 4 for field in domain_fields):
        raise SystemExit(f"FAIL: {path} has a non-four-domain manifest field")
    residual_tolerance = float(manifest["solver_controls"]["NLtol"])
    for domain, history in enumerate(manifest["newton_history"]):
        history_values = np.asarray(history, dtype=float)
        if history_values.ndim != 2 or history_values.shape[0] == 0 or history_values.shape[1] != 4:
            raise SystemExit(f"FAIL: {path} domain {domain + 1} Newton history is not nonempty Nx4")
        if float(manifest["final_residual"][domain]) != float(history_values[-1, 3]):
            raise SystemExit(f"FAIL: {path} domain {domain + 1} final residual disagrees with history")
        if int(manifest["newton_iterations"][domain]) != int(history_values[-1, 2]):
            raise SystemExit(f"FAIL: {path} domain {domain + 1} Newton count disagrees with history")
        if history_values[-1, 3] > residual_tolerance:
            raise SystemExit(f"FAIL: {path} domain {domain + 1} did not reach NLtol")
    expected_linear_solves = sum(int(value) for value in manifest["newton_iterations"])
    if len(manifest["gmres_iterations"]) != expected_linear_solves:
        raise SystemExit(
            f"FAIL: {path} has {len(manifest['gmres_iterations'])} GMRES records, "
            f"expected {expected_linear_solves}")
    for key in ("maximum_conditions", "max_conditions", "maximum_condition_by_domain"):
        if key in manifest and manifest[key] != [1, 2, 1, 2]:
            raise SystemExit(f"FAIL: {path} {key} differs from [1,2,1,2]")
    dimension = int(manifest["dimension"])
    solution = np.fromfile(path / "phase5_solution.bin", dtype=np.float64)
    coordinates = np.fromfile(path / "phase5_coordinates.bin", dtype=np.float64)
    centroids = np.fromfile(path / "phase5_element_centroids.bin", dtype=np.float64)
    if coordinates.size != dimension * solution.size or centroids.size != coordinates.size:
        raise SystemExit(f"FAIL: inconsistent solution/coordinate sizes in {path}")
    numeric = [solution, coordinates, centroids]
    numeric.extend(np.asarray(manifest[field], dtype=float) for field in measured_fields)
    numeric.extend(np.asarray(history, dtype=float) for history in manifest["newton_history"])
    if any(not np.isfinite(values).all() for values in numeric):
        raise SystemExit(f"FAIL: NaN or Inf in {path}")
    return manifest, solution, coordinates.reshape((-1, dimension)), centroids.reshape((-1, dimension))


def domain_sizes(manifest):
    p = int(manifest["porder"])
    dim = int(manifest["dimension"])
    sizes = []
    for family, elements in zip(manifest["families"], manifest["element_counts"]):
        if int(family) == 1:
            npe = (p + 1) ** dim
        elif dim == 2:
            npe = (p + 1) * (p + 2) // 2
        else:
            npe = (p + 1) * (p + 2) * (p + 3) // 6
        sizes.append(npe * int(elements))
    return sizes


def canonical_blocks(manifest, solution, coordinates, centroids):
    sizes = domain_sizes(manifest)
    if sum(sizes) != solution.size:
        raise SystemExit(
            f"FAIL: manifest implies {sum(sizes)} scalar entries, file has {solution.size}")
    blocks = []
    offset = 0
    for domain, count in enumerate(sizes, start=1):
        stop = offset + count
        keys = np.hstack((centroids[offset:stop], coordinates[offset:stop]))
        order = np.lexsort(tuple(keys[:, column] for column in range(keys.shape[1] - 1, -1, -1)))
        blocks.append((domain, keys[order], solution[offset:stop][order]))
        offset = stop
    return blocks


def compare_keys(reference, candidate, label):
    if reference.shape != candidate.shape:
        raise SystemExit(f"FAIL: physical-key shape differs for {label}")
    if np.array_equal(reference, candidate):
        bound = 0.0
    else:
        scale = max(1.0, float(np.max(np.abs(reference))), float(np.max(np.abs(candidate))))
        bound = 64.0 * np.finfo(np.float64).eps * scale
        difference = np.max(np.abs(reference - candidate), axis=1)
        if np.any(difference > bound):
            raise SystemExit(
                f"FAIL: unmatched physical coordinate for {label}: "
                f"max={float(np.max(difference)):.17e}, bound={bound:.17e}")
    if reference.shape[0] > 1:
        separation = np.max(np.abs(np.diff(reference, axis=0)), axis=1)
        if np.any(separation <= bound):
            raise SystemExit(f"FAIL: ambiguous physical key for {label}")
    return bound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("triples", nargs="+")
    args = parser.parse_args()
    if len(args.triples) % 3:
        raise SystemExit("provide MATLAB, Python, Julia result directories per case")

    for first in range(0, len(args.triples), 3):
        paths = [pathlib.Path(value) for value in args.triples[first:first + 3]]
        results = [load_result(path) for path in paths]
        manifests = [result[0] for result in results]
        reference = manifests[0]
        for field in EXACT_FIELDS:
            if any(manifest[field] != reference[field] for manifest in manifests[1:]):
                raise SystemExit(f"FAIL: exact structural field {field} differs")
        lengths = [[len(history) for history in manifest["newton_history"]]
                   for manifest in manifests]
        if any(value != lengths[0] for value in lengths[1:]):
            raise SystemExit("FAIL: Newton history lengths differ")

        blocks = [canonical_blocks(*result) for result in results]
        solution_differences = []
        coordinate_bounds = []
        for language in range(1, 3):
            for ref_block, candidate_block in zip(blocks[0], blocks[language]):
                domain = ref_block[0]
                coordinate_bounds.append(compare_keys(
                    ref_block[1], candidate_block[1],
                    f"{reference['case']} domain {domain} frontend {language}"))
                difference = np.abs(ref_block[2] - candidate_block[2])
                scale = max(1.0, float(np.max(np.abs(ref_block[2]))))
                solution_differences.append((language, domain, float(np.max(difference)), scale))

        for language, domain, difference, scale in solution_differences:
            print(f"MEASURE frontend case={reference['case']} candidate={language} "
                  f"domain={domain} solution_max_abs={difference:.17e} scale={scale:.17e}")
        for language in range(1, 3):
            for field in ("per_domain_l2", "combined_l2", "max_nodal_error",
                          "final_residual"):
                ref_values = np.asarray(manifests[0][field], dtype=float)
                candidate_values = np.asarray(manifests[language][field], dtype=float)
                if ref_values.shape != candidate_values.shape:
                    raise SystemExit(
                        f"FAIL: {field} shape differs for candidate {language}")
                difference = float(np.max(np.abs(ref_values - candidate_values)))
                scale = max(1.0, float(np.max(np.abs(ref_values))))
                print(f"MEASURE frontend case={reference['case']} candidate={language} "
                      f"field={field} max_abs={difference:.17e} scale={scale:.17e}")
            for domain, (ref_history, candidate_history) in enumerate(zip(
                    manifests[0]["newton_history"], manifests[language]["newton_history"]), start=1):
                ref_values = np.asarray(ref_history, dtype=float)
                candidate_values = np.asarray(candidate_history, dtype=float)
                if ref_values.shape != candidate_values.shape:
                    raise SystemExit(
                        f"FAIL: Newton history shape differs for candidate {language} domain {domain}")
                difference = float(np.max(np.abs(ref_values - candidate_values)))
                scale = max(1.0, float(np.max(np.abs(ref_values))))
                print(f"MEASURE frontend case={reference['case']} candidate={language} "
                      f"domain={domain} history_max_abs={difference:.17e} scale={scale:.17e}")
        print(f"STRUCTURE PASS case={reference['case']} coordinate_bound_max="
              f"{max(coordinate_bounds, default=0.0):.17e}; numerical values are measurement-only")


if __name__ == "__main__":
    main()
