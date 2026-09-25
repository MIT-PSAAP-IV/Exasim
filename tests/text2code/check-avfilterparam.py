#!/usr/bin/env python3
"""Validate the AV-filter block in an Exasim app.bin file."""

import array
import math
import sys


def read_avfilterparam(filename):
    values = array.array("d")
    with open(filename, "rb") as stream:
        stream.seek(0, 2)
        count = stream.tell() // values.itemsize
        stream.seek(0)
        values.fromfile(stream, count)

    nsize_length = int(values[0])
    nsize = [int(value) for value in values[1 : 1 + nsize_length]]
    if nsize_length <= 19:
        raise RuntimeError(f"{filename}: app.bin has no nsize[19] slot")
    if nsize[19] != 2:
        raise RuntimeError(f"{filename}: expected nsize[19] == 2, got {nsize[19]}")

    offset = 1 + nsize_length + sum(nsize[:19])
    return tuple(values[offset : offset + nsize[19]])


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit(
            "usage: check-avfilterparam.py APP_BIN METHOD COEFFICIENT [REFERENCE_APP_BIN]"
        )

    actual = read_avfilterparam(sys.argv[1])
    expected = (float(sys.argv[2]), float(sys.argv[3]))
    if actual[0] != expected[0] or not math.isclose(
        actual[1], expected[1], rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise RuntimeError(f"{sys.argv[1]}: expected {expected}, got {actual}")

    if len(sys.argv) == 5:
        reference = read_avfilterparam(sys.argv[4])
        if actual != reference:
            raise RuntimeError(
                f"AV-filter mismatch: Text2Code {actual}, native preprocessing {reference}"
            )

    print(f"{sys.argv[1]}: nsize[19] = 2, avfilterparam = {actual}")


if __name__ == "__main__":
    main()
