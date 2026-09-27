"""Compare independently partitioned Text2Code and MATLAB runs."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


NPE = 9
ND = 2
NRANKS = 4


def read_doubles(path: Path, header: bool = False) -> list[float]:
    data = [value[0] for value in struct.iter_unpack("=d", path.read_bytes())]
    return data[3:] if header else data


def owned_counts(directory: Path) -> list[int]:
    counts = []
    for rank in range(NRANKS):
        size = (directory / f"outxdg_np{rank}.bin").stat().st_size // 8
        if size % (NPE * ND):
            raise ValueError(f"Invalid outxdg size on rank {rank}: {size}")
        counts.append(size // (NPE * ND))
    return counts


def load_owned(
    directory: Path,
    stem: str,
    components: int,
    counts: list[int],
    *,
    header: bool = False,
) -> list[list[float]]:
    block_size = NPE * components
    elements: list[list[float]] = []
    for rank, owned in enumerate(counts):
        values = read_doubles(directory / f"{stem}_np{rank}.bin", header)
        if len(values) % block_size:
            raise ValueError(f"Invalid field size: {stem}_np{rank}.bin")
        total = len(values) // block_size
        if total < owned:
            raise ValueError(f"{stem}_np{rank}.bin has fewer than {owned} elements")
        for element in range(owned):
            start = element * block_size
            elements.append(values[start : start + block_size])
    return elements


def coordinates(
    directory: Path, iteration: int, counts: list[int]
) -> list[list[float]]:
    if iteration < 10:
        return load_owned(
            directory, f"out_meshadapt_aviter{iteration}_iter1_xdg", ND, counts
        )
    return load_owned(directory, "outxdg", ND, counts)


def centroids(elements: list[list[float]]) -> list[tuple[float, float]]:
    return [
        (sum(element[:NPE]) / NPE, sum(element[NPE : 2 * NPE]) / NPE)
        for element in elements
    ]


def element_map(
    reference_coordinates: list[list[float]], candidate_coordinates: list[list[float]]
) -> tuple[list[int], float]:
    reference_centroids = centroids(reference_coordinates)
    candidate_centroids = centroids(candidate_coordinates)
    if len(reference_centroids) != len(candidate_centroids):
        raise ValueError("Reference and candidate have different global element counts")

    mapping = []
    maximum_distance = 0.0
    for x, y in candidate_centroids:
        nearest = min(
            range(len(reference_centroids)),
            key=lambda index: (reference_centroids[index][0] - x) ** 2
            + (reference_centroids[index][1] - y) ** 2,
        )
        distance = math.hypot(
            reference_centroids[nearest][0] - x,
            reference_centroids[nearest][1] - y,
        )
        mapping.append(nearest)
        maximum_distance = max(maximum_distance, distance)
    if len(set(mapping)) != len(mapping):
        raise ValueError("Coordinate matching is not one-to-one")
    return mapping, maximum_distance


def compare_elements(
    reference: list[list[float]], candidate: list[list[float]], mapping: list[int]
) -> tuple[float, float]:
    sum_difference_squared = 0.0
    sum_reference_squared = 0.0
    maximum_absolute = 0.0
    for candidate_index, reference_index in enumerate(mapping):
        expected = reference[reference_index]
        actual = candidate[candidate_index]
        if len(expected) != len(actual):
            raise ValueError("Reference and candidate element blocks differ in size")
        for expected_value, actual_value in zip(expected, actual):
            difference = actual_value - expected_value
            sum_difference_squared += difference * difference
            sum_reference_squared += expected_value * expected_value
            maximum_absolute = max(maximum_absolute, abs(difference))
    denominator = max(math.sqrt(sum_reference_squared), float.fromhex("0x1p-1022"))
    return math.sqrt(sum_difference_squared) / denominator, maximum_absolute


def field_stem(iteration: int, suffix: str) -> str:
    return f"out_meshadapt_aviter{iteration}_{suffix}"


def compare(reference_directory: Path, candidate_directory: Path) -> None:
    reference_counts = owned_counts(reference_directory)
    candidate_counts = owned_counts(candidate_directory)
    print(f"MATLAB owned elements:    {reference_counts}")
    print(f"Text2Code owned elements: {candidate_counts}")
    print("Errors are relative L2 / maximum absolute; no tolerance is enforced.\n")

    fields = (
        ("wall_distance", 1),
        ("av_raw", 1),
        ("av_smoothed", 1),
        ("flow_solution", 12),
        ("field1", 1),
        ("field2", 1),
        ("eta", 1),
        ("iter1_displacement", 2),
        ("iter1_xdg", 2),
    )
    for iteration in range(1, 11):
        reference_coordinates = coordinates(reference_directory, iteration, reference_counts)
        candidate_coordinates = coordinates(candidate_directory, iteration, candidate_counts)
        mapping, centroid_error = element_map(reference_coordinates, candidate_coordinates)
        print(f"AV iteration {iteration} (maximum centroid match distance {centroid_error:.6e})")
        for suffix, components in fields:
            if iteration == 10 and suffix in {
                "field1",
                "field2",
                "eta",
                "iter1_displacement",
                "iter1_xdg",
            }:
                continue
            reference = load_owned(
                reference_directory, field_stem(iteration, suffix), components, reference_counts
            )
            candidate = load_owned(
                candidate_directory, field_stem(iteration, suffix), components, candidate_counts
            )
            relative, absolute = compare_elements(reference, candidate, mapping)
            print(f"  {suffix:20s} {relative:.6e} / {absolute:.6e}")

    final_mapping, centroid_error = element_map(
        coordinates(reference_directory, 10, reference_counts),
        coordinates(candidate_directory, 10, candidate_counts),
    )
    print(f"\nFinal fields (maximum centroid match distance {centroid_error:.6e})")
    for stem, components, header in (
        ("outxdg", 2, False),
        ("outudg", 12, True),
        ("outvdg", 2, False),
        ("outwdg", 15, True),
    ):
        reference = load_owned(
            reference_directory, stem, components, reference_counts, header=header
        )
        candidate = load_owned(
            candidate_directory, stem, components, candidate_counts, header=header
        )
        relative, absolute = compare_elements(reference, candidate, final_mapping)
        print(f"  {stem:20s} {relative:.6e} / {absolute:.6e}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=here.parents[2]
        / "examples"
        / "MeshAdaptivity"
        / "cylindermach8_equichem"
        / "dataout",
    )
    parser.add_argument("--candidate", type=Path, default=here / "dataout")
    arguments = parser.parse_args()
    compare(arguments.reference, arguments.candidate)


if __name__ == "__main__":
    main()
