"""Compare MATLAB, Python, and Julia heat-transfer example outputs.

The comparison removes Exasim's three-value output header and matches owned
elements by physical centroids.  This avoids treating harmless partition and
local-node permutations as solution errors.  Independently generated Gmsh
meshes are summarized by mesh-independent field statistics when their nodes do
not coincide closely enough for a pointwise comparison.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


CASES = {
    "Isoq2D_matching": ((4, 9), (2, 6)),
    "Isoq2D_nonmatching": ((4, 9), (2, 6)),
    "cylindermach8_matching": ((2, 25), (2, 25)),
    "cylindermach8_nonmatching": ((2, 25), (2, 25)),
    "cylindermach8_quad_tri": ((2, 25), (2, 15)),
}

# Native symbolic engines may reorder round-off-level arithmetic, but all
# frontends must converge to the same discrete solution.
POINTWISE_RELATIVE_TOLERANCE = 1.0e-6


def _read_xdg(filename: Path, npe: int) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.float64)
    nsize_length = int(data[0])
    nsize = data[1 : 1 + nsize_length].astype(np.int64)
    offset = 1 + nsize_length + int(nsize[0])
    return data[offset : offset + int(nsize[1])].reshape((npe, 2, -1), order="F")


def _read_output(filename: Path) -> np.ndarray:
    data = np.fromfile(filename, dtype=np.float64)
    if data.size < 3:
        raise ValueError(f"Incomplete Exasim output: {filename}")
    npe, nc, ne = (int(value) for value in data[:3])
    expected = 3 + npe * nc * ne
    if data.size != expected:
        raise ValueError(f"Malformed Exasim output: {filename}")
    return data[3:].reshape((npe, nc, ne), order="F")


def _load(datain: Path, dataout: Path, ranks: int, npe: int):
    coordinates, solution = [], []
    for rank in range(ranks):
        values = _read_output(dataout / f"outudg_np{rank}.bin")
        nodes = _read_xdg(datain / f"sol{rank + 1}.bin", npe)
        # Partitioned sol files append ghost elements; backend output contains
        # only the owned prefix.
        coordinates.append(nodes[:, :, : values.shape[2]])
        solution.append(values)
    return np.concatenate(coordinates, axis=2), np.concatenate(solution, axis=2)


def _paths(case: Path, frontend: str, model: int):
    if frontend == "MATLAB":
        if case.name == "Isoq2D_nonmatching":
            suffix = "" if model == 1 else "1"
            return case / f"datain{suffix}", case / f"dataout{suffix}"
        return case / f"datain{model}", case / f"dataout{model}"
    root = case / f"{frontend.lower()}_run" / "coupled"
    suffix = Path() if model == 1 else Path("1")
    return root / "datain" / suffix, root / "dataout" / suffix


def _directory_digest(directory: Path):
    digest = hashlib.sha256()
    files = sorted(path for path in directory.iterdir() if path.is_file())
    for path in files:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest(), len(files)


def _verify_isoq_native_artifacts(case: Path) -> int:
    """Require the native ISOQ runs to consume identical discrete problems."""
    failures = 0
    run_roots = {
        "MATLAB": case,
        "Python": case / "python_run" / "coupled",
        "Julia": case / "julia_run" / "coupled",
    }
    for model, (ranks, _) in enumerate(CASES[case.name], start=1):
        names = ["app.bin", "master.bin"]
        names += [f"mesh{rank}.bin" for rank in range(1, ranks + 1)]
        names += [f"sol{rank}.bin" for rank in range(1, ranks + 1)]
        hashes = {}
        for frontend in run_roots:
            datain, _ = _paths(case, frontend, model)
            missing = [name for name in names if not (datain / name).is_file()]
            if missing:
                print(f"    model {model} {frontend}: missing {', '.join(missing)}")
                failures += 1
                continue
            hashes[frontend] = [hashlib.sha256((datain / name).read_bytes()).digest()
                                for name in names]
        if len(hashes) == 3 and not (
            hashes["MATLAB"] == hashes["Python"] == hashes["Julia"]
        ):
            print(f"    model {model}: native input binaries differ")
            failures += 1

    kernel_directories = {
        "MATLAB": [case / ".exasim" / "kernels", case / ".exasim" / "models" / "1" / "kernels"],
        "Python": [run_roots["Python"] / ".exasim" / "kernels", run_roots["Python"] / ".exasim" / "models" / "1" / "kernels"],
        "Julia": [run_roots["Julia"] / ".exasim" / "kernels", run_roots["Julia"] / ".exasim" / "models" / "1" / "kernels"],
    }
    for model in range(2):
        try:
            digests = {frontend: _directory_digest(paths[model])
                       for frontend, paths in kernel_directories.items()}
        except (FileNotFoundError, NotADirectoryError) as error:
            print(f"    model {model + 1}: missing kernel directory: {error}")
            failures += 1
            continue
        if len(set(digests.values())) != 1:
            print(f"    model {model + 1}: generated Kokkos sources differ")
            failures += 1
    if failures == 0:
        print("  native input binaries and generated Kokkos sources: byte-identical")
    return failures


def _pointwise_error(candidate, reference, coordinate_tolerance=2.0e-8):
    x_candidate, u_candidate = candidate
    x_reference, u_reference = reference
    if x_candidate.shape[2] != x_reference.shape[2]:
        return None

    # Centroids are separated by much more than the matching tolerance. Spatial
    # buckets keep the matching linear-time without a rounding-boundary failure.
    bucket_width = 4.0 * coordinate_tolerance
    centroid_map = {}
    for element, centroid in enumerate(x_reference.mean(axis=0).T):
        key = tuple(np.floor(centroid / bucket_width).astype(np.int64))
        centroid_map.setdefault(key, []).append(element)

    differences, references = [], []
    max_coordinate_error = 0.0
    used = set()
    for element, centroid in enumerate(x_candidate.mean(axis=0).T):
        key = np.floor(centroid / bucket_width).astype(np.int64)
        candidates = []
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                candidates.extend(
                    centroid_map.get((key[0] + offset_x, key[1] + offset_y), [])
                )
        candidates = [candidate for candidate in candidates if candidate not in used]
        if not candidates:
            return None
        errors = [
            np.linalg.norm(x_reference[:, :, donor].mean(axis=0) - centroid)
            for donor in candidates
        ]
        donor = candidates[int(np.argmin(errors))]
        if errors[int(np.argmin(errors))] > coordinate_tolerance:
            return None
        used.add(donor)
        reference_nodes = x_reference[:, :, donor]
        distances = np.linalg.norm(
            x_candidate[:, :, element, None] - reference_nodes.T[None, :, :], axis=1
        )
        permutation = np.argmin(distances, axis=1)
        node_error = distances[np.arange(distances.shape[0]), permutation]
        max_coordinate_error = max(max_coordinate_error, float(node_error.max()))
        if max_coordinate_error > coordinate_tolerance or len(set(permutation)) != len(permutation):
            return None
        differences.append((u_candidate[:, :, element] - u_reference[permutation, :, donor]).ravel())
        references.append(u_reference[permutation, :, donor].ravel())

    difference = np.concatenate(differences)
    baseline = np.concatenate(references)
    relative = np.linalg.norm(difference) / max(
        np.linalg.norm(baseline), np.finfo(np.float64).eps
    )
    return relative, float(np.max(np.abs(difference))), max_coordinate_error


def _statistics(values: np.ndarray):
    flat = values.ravel()
    return np.linalg.norm(flat) / np.sqrt(flat.size), float(flat.min()), float(flat.max())


def verify_case(case: Path) -> int:
    specifications = CASES[case.name]
    print(case.name)
    failures = 0
    if case.name == "Isoq2D_nonmatching":
        failures += _verify_isoq_native_artifacts(case)
    for model, (ranks, npe) in enumerate(specifications, start=1):
        fields = {}
        for frontend in ("MATLAB", "Python", "Julia"):
            datain, dataout = _paths(case, frontend, model)
            fields[frontend] = _load(datain, dataout, ranks, npe)
        print(f"  model {model}")
        for frontend in ("Python", "Julia"):
            error = _pointwise_error(fields[frontend], fields["MATLAB"])
            if error is not None:
                relative, maximum, geometry = error
                print(
                    f"    {frontend:6s}: relative L2={relative:.6e}, "
                    f"max abs={maximum:.6e}, geometry={geometry:.3e}"
                )
                if relative > POINTWISE_RELATIVE_TOLERANCE:
                    failures += 1
            else:
                candidate = _statistics(fields[frontend][1])
                reference = _statistics(fields["MATLAB"][1])
                norm_error = abs(candidate[0] - reference[0]) / max(
                    abs(reference[0]), np.finfo(np.float64).eps
                )
                print(
                    f"    {frontend:6s}: independent mesh; RMS relative={norm_error:.6e}, "
                    f"range=[{candidate[1]:.6e}, {candidate[2]:.6e}] "
                    f"(MATLAB [{reference[1]:.6e}, {reference[2]:.6e}])"
                )
                if norm_error > 1.0e-2:
                    failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", choices=sorted(CASES))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cases = args.cases or list(CASES)
    failures = sum(verify_case(root / name) for name in cases)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
