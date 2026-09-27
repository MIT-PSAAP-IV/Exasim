"""Compare this Text2Code result and partition with the MATLAB reference."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


NPE = 9
ND = 2
NRANKS = 8


def read_doubles(path: Path, header: bool = False) -> list[float]:
    values = [item[0] for item in struct.iter_unpack("=d", path.read_bytes())]
    return values[3:] if header else values


def owned_counts(directory: Path) -> list[int]:
    counts = []
    for rank in range(NRANKS):
        entries = (directory / f"outxdg_np{rank}.bin").stat().st_size // 8
        if entries % (NPE * ND):
            raise ValueError(f"Invalid outxdg size on rank {rank}: {entries}")
        counts.append(entries // (NPE * ND))
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
    elements = []
    for rank, owned in enumerate(counts):
        values = read_doubles(directory / f"{stem}_np{rank}.bin", header)
        if len(values) % block_size:
            raise ValueError(f"Invalid field size: {stem}_np{rank}.bin")
        if len(values) // block_size < owned:
            raise ValueError(f"{stem}_np{rank}.bin has fewer than {owned} elements")
        for element in range(owned):
            start = element * block_size
            elements.append(values[start : start + block_size])
    return elements


def read_element_partition(path: Path) -> tuple[list[int], int]:
    values = read_doubles(path)
    number_of_sizes = int(values[0])
    sizes = [int(value) for value in values[1 : 1 + number_of_sizes]]
    position = 1 + number_of_sizes + sizes[0] + sum(sizes[1:9])
    elements = [int(value) for value in values[position : position + sizes[9]]]
    partition_points = position + sizes[9]
    owned = sum(int(value) for value in values[partition_points : partition_points + 2])
    return elements, owned


def global_element_ids(mesh_directory: Path, counts: list[int]) -> list[int]:
    ids = []
    for rank, expected_owned in enumerate(counts):
        partition, owned = read_element_partition(mesh_directory / f"mesh{rank + 1}.bin")
        if owned != expected_owned:
            raise ValueError(
                f"Rank {rank} mesh owns {owned} elements, output has {expected_owned}"
            )
        ids.extend(partition[:owned])

    if ids and min(ids) == 1:
        ids = [element - 1 for element in ids]
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("Owned global element IDs do not cover the global mesh")
    return ids


def owner_map(mesh_directory: Path, counts: list[int]) -> list[int]:
    ids = global_element_ids(mesh_directory, counts)
    owners = [0] * len(ids)
    offset = 0
    for rank, owned in enumerate(counts, start=1):
        for element in ids[offset : offset + owned]:
            owners[element] = rank
        offset += owned
    return owners


def reorder(elements: list[list[float]], ids: list[int]) -> list[list[float]]:
    return [block for _, block in sorted(zip(ids, elements), key=lambda item: item[0])]


def errors(
    reference: list[list[float]], candidate: list[list[float]]
) -> tuple[float, float]:
    difference_squared = 0.0
    reference_squared = 0.0
    maximum_absolute = 0.0
    if len(reference) != len(candidate):
        raise ValueError("Reference and candidate have different element counts")
    for expected, actual in zip(reference, candidate):
        if len(expected) != len(actual):
            raise ValueError("Reference and candidate element blocks differ in size")
        for expected_value, actual_value in zip(expected, actual):
            difference = actual_value - expected_value
            difference_squared += difference * difference
            reference_squared += expected_value * expected_value
            maximum_absolute = max(maximum_absolute, abs(difference))
    denominator = max(math.sqrt(reference_squared), float.fromhex("0x1p-1022"))
    return math.sqrt(difference_squared) / denominator, maximum_absolute


def compare(reference_directory: Path, candidate_directory: Path, partition: Path) -> None:
    reference_counts = owned_counts(reference_directory)
    candidate_counts = owned_counts(candidate_directory)
    reference_mesh = reference_directory.parent / "datain"
    candidate_mesh = candidate_directory.parent / "datain"
    reference_ids = global_element_ids(reference_mesh, reference_counts)
    candidate_ids = global_element_ids(candidate_mesh, candidate_counts)

    requested_owners = [int(value) for value in read_doubles(partition)]
    reference_owners = owner_map(reference_mesh, reference_counts)
    if requested_owners != reference_owners:
        raise ValueError("partition.bin does not match the MATLAB element ownership")
    candidate_owners = owner_map(candidate_mesh, candidate_counts)
    if candidate_owners != reference_owners:
        raise ValueError("Text2Code preprocessing did not preserve partition.bin ownership")

    print(f"MATLAB owned elements:    {reference_counts}")
    print(f"Text2Code owned elements: {candidate_counts}")
    print("partition.bin: exact MATLAB ownership match")
    print("Errors are relative L2 / maximum absolute:")

    for stem, components, header in (
        ("outxdg", 2, False),
        ("outudg", 24, True),
        ("outvdg", 2, False),
        ("outwdg", 1, True),
    ):
        reference = reorder(
            load_owned(reference_directory, stem, components, reference_counts, header=header),
            reference_ids,
        )
        candidate = reorder(
            load_owned(candidate_directory, stem, components, candidate_counts, header=header),
            candidate_ids,
        )
        relative, absolute = errors(reference, candidate)
        print(f"  {stem:10s} {relative:.6e} / {absolute:.6e}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=here.parents[2]
        / "examples"
        / "MeshAdaptivity"
        / "isoq2d_nonequichem"
        / "dataout",
    )
    parser.add_argument("--candidate", type=Path, default=here / "dataout")
    parser.add_argument("--partition", type=Path, default=here / "partition.bin")
    arguments = parser.parse_args()
    compare(arguments.reference, arguments.candidate, arguments.partition)


if __name__ == "__main__":
    main()
