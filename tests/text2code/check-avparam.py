#!/usr/bin/env python3
"""Validate the interleaved AV-continuation block in an Exasim app.bin."""

import array
import math
import sys


def read_avparam(filename):
    values = array.array("d")
    with open(filename, "rb") as stream:
        stream.seek(0, 2)
        count = stream.tell() // values.itemsize
        stream.seek(0)
        values.fromfile(stream, count)

    nsize_length = int(values[0])
    nsize = [int(value) for value in values[1 : 1 + nsize_length]]
    if nsize_length <= 15:
        raise RuntimeError(f"{filename}: app.bin has no nsize[15] slot")

    offset = 1 + nsize_length + sum(nsize[:15])
    return list(values[offset : offset + nsize[15]])


def generated_schedule(iterations, alpha, coeff_start, coeff_end):
    if iterations < 2:
        return None
    denominator = math.expm1(alpha) if abs(alpha) > 1.0e-14 else 1.0
    result = []
    for index in range(iterations):
        t = index / (iterations - 1)
        if abs(alpha) <= 1.0e-14:
            first, second = 1.0 - t, t
        else:
            first = math.expm1(alpha * (1.0 - t)) / denominator
            second = math.expm1(alpha * t) / denominator
        result.extend((coeff_start * first, coeff_end * second))
    result[0:2] = [coeff_start, 0.0]
    result[-2:] = [0.0, coeff_end]
    return result


def assert_close(actual, expected, label):
    if len(actual) != len(expected) or any(
        not math.isclose(a, e, rel_tol=1.0e-14, abs_tol=1.0e-15)
        for a, e in zip(actual, expected)
    ):
        raise RuntimeError(f"{label}: expected {expected}, got {actual}")


def main():
    if len(sys.argv) < 6:
        raise SystemExit(
            "usage: check-avparam.py APP_BIN ITER LOG_SCALE START END "
            "[REFERENCE_APP_BIN] [--expected comma-separated-values]"
        )

    filename = sys.argv[1]
    iterations = int(sys.argv[2])
    expected = generated_schedule(
        iterations, float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    )
    remaining = sys.argv[6:]
    if "--expected" in remaining:
        index = remaining.index("--expected")
        expected = [float(value) for value in remaining[index + 1].split(",")]
        del remaining[index : index + 2]
    if expected is None:
        raise RuntimeError("ITER < 2 requires --expected")

    actual = read_avparam(filename)
    assert_close(actual, expected, filename)
    if remaining:
        reference = read_avparam(remaining[0])
        assert_close(actual, reference, "Text2Code/native AV-continuation mismatch")

    print(f"{filename}: nsize[15] = {len(actual)}, avparam = {actual}")


if __name__ == "__main__":
    main()
