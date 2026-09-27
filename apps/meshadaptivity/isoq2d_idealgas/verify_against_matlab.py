"""Compare the Text2Code ISOQ result with the MATLAB reference result."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


NPE = 9
ND = 2
NRANKS = 4


def read_doubles(path: Path, header: bool = False) -> list[float]:
    values = [item[0] for item in struct.iter_unpack("=d", path.read_bytes())]
    return values[3:] if header else values


def owned_counts(directory: Path) -> list[int]:
    counts = []
    for rank in range(NRANKS):
        count = (directory / f"outxdg_np{rank}.bin").stat().st_size // 8
        if count % (NPE * ND):
            raise ValueError(f"Invalid outxdg size on rank {rank}: {count}")
        counts.append(count // (NPE * ND))
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


def read_element_partition(path: Path) -> list[int]:
    values = read_doubles(path)
    number_of_sizes = int(values[0])
    sizes = [int(value) for value in values[1 : 1 + number_of_sizes]]
    position = 1 + number_of_sizes + sizes[0] + sum(sizes[1:9])
    return [int(value) for value in values[position : position + sizes[9]]]


def global_element_ids(mesh_directory: Path, counts: list[int]) -> list[int]:
    ids = []
    for rank, owned in enumerate(counts):
        partition = read_element_partition(mesh_directory / f"mesh{rank + 1}.bin")
        if len(partition) < owned:
            raise ValueError(f"mesh{rank + 1}.bin has fewer than {owned} elements")
        ids.extend(partition[:owned])

    # MATLAB stores one-based global IDs; text2code stores zero-based IDs.
    if ids and min(ids) == 1:
        ids = [element - 1 for element in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("Owned global element IDs are not unique")
    return ids


def reorder_by_global_id(
    elements: list[list[float]], ids: list[int]
) -> list[list[float]]:
    if len(elements) != len(ids):
        raise ValueError("Field and global-element ID counts differ")
    ordered = sorted(zip(ids, elements), key=lambda item: item[0])
    if [item[0] for item in ordered] != list(range(len(ordered))):
        raise ValueError("Owned global element IDs do not cover the global mesh")
    return [item[1] for item in ordered]


def compare_elements(
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


def compare(reference_directory: Path, candidate_directory: Path) -> None:
    reference_counts = owned_counts(reference_directory)
    candidate_counts = owned_counts(candidate_directory)
    reference_ids = global_element_ids(reference_directory.parent / "datain", reference_counts)
    candidate_ids = global_element_ids(candidate_directory.parent / "datain", candidate_counts)
    print(f"MATLAB owned elements:    {reference_counts}")
    print(f"Text2Code owned elements: {candidate_counts}")
    print("Fields are reordered by global element ID from each mesh partition.")
    print("Errors are relative L2 / maximum absolute:")

    for stem, components, header in (
        ("outxdg", 2, False),
        ("outvdg", 2, False),
        ("outudg", 12, True),
    ):
        reference = reorder_by_global_id(
            load_owned(
                reference_directory, stem, components, reference_counts, header=header
            ),
            reference_ids,
        )
        candidate = reorder_by_global_id(
            load_owned(
                candidate_directory, stem, components, candidate_counts, header=header
            ),
            candidate_ids,
        )
        relative, absolute = compare_elements(reference, candidate)
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
        / "isoq2d_idealgas"
        / "dataout",
    )
    parser.add_argument("--candidate", type=Path, default=here / "dataout")
    arguments = parser.parse_args()
    compare(arguments.reference, arguments.candidate)


if __name__ == "__main__":
    main()
