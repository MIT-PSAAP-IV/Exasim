#!/usr/bin/env python3
"""Cross-check for the surface-vis consumer (stdlib only: no numpy on CI runners).

Usage: check_surf_vtu.py <datadir>   (single rank, NP=1)

The fixture runs saveParaview + saveSolBouFreq with ibs = 1 and saveSolBouLoc = 0,
so both outbousurf_np0.bin and outsurf.vtu hold the SAME nodal SurfaceQuantities
evaluation: the .bin straight from evalSurfaceQuantities, the VTU after the 1:1
DG scatter onto the ParaView sub-cell mesh (surf node = face node in face order).
Every VTU point value must therefore match the .bin value up to the float32 cast
(tolerance 1e-6). A mismatch means the ParaView path evaluated or scattered
different data than the boundary writer (e.g. a Gauss-vs-nodes mix-up).
"""
import glob
import re
import struct
import sys


def read_bin(path):
    with open(path, "rb") as f:
        blob = f.read()
    h = struct.unpack("<3d", blob[:24])
    return h, blob[24:]


def parse_vtu(path):
    with open(path, "rb") as f:
        blob = f.read()
    head, sep, app = blob.partition(b"<AppendedData")
    assert sep, "no AppendedData section"
    m = re.search(rb'NumberOfPoints="(\d+)"', head)
    assert m, "no NumberOfPoints"
    npoints = int(m.group(1))
    pdata = head.split(b"<PointData")[1].split(b"</PointData>")[0]
    arrays = re.findall(
        rb'<DataArray type="Float32" Name="([^"]+)" Format="appended" offset="(\d+)"',
        pdata,
    )
    assert arrays, "no appended Float32 point arrays"
    us = app.index(b"_")
    start = us + 1
    out = []
    for name, off in arrays:
        pos = start + int(off)
        (nbytes,) = struct.unpack("<Q", app[pos : pos + 8])
        vals = struct.unpack("<%df" % (nbytes // 4), app[pos + 8 : pos + 8 + nbytes])
        assert len(vals) == npoints, (name, len(vals), npoints)
        out.append((name.decode(), vals))
    return npoints, out


def main():
    data = sys.argv[1]
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    (np_, nfbou, nsq), payload = read_bin(data + "/outbousurf_np%d.bin" % rank)
    np_, nfbou, nsq = int(np_), int(nfbou), int(nsq)
    nper = np_ * nfbou * nsq
    nvals = len(payload) // 8
    assert nvals >= nper and nvals % nper == 0, (nvals, nper)
    nsteps = nvals // nper
    step = struct.unpack("<%dd" % nper, payload[: 8 * nper])

    tagged = glob.glob(data + "/outsurf_%05d.vtu" % rank)
    if tagged:
        vtu = sorted(tagged)
    else:
        assert rank == 0, "no outsurf piece found for rank %d" % rank
        vtu = sorted(
            glob.glob(data + "/outsurf.vtu") + glob.glob(data + "/outsurf.pvtu")
        )
    assert vtu, "no outsurf piece found for rank %d" % rank
    npoints, arrays = parse_vtu(vtu[0])
    assert len(arrays) == nsq, (len(arrays), nsq)
    assert npoints == np_ * nfbou, (npoints, np_, nfbou)

    worst = 0.0
    for k, (name, vals) in enumerate(arrays):
        base = k * np_ * nfbou
        for i, v in enumerate(vals):
            ref = step[base + i]
            d = abs(v - ref)
            if d > worst:
                worst = d
            tol = 1e-6 * max(1.0, abs(ref))
            assert d <= tol, (name, i, v, ref, d)
    print("vtu-vs-outbousurf ok: %d points x %d fields, max diff %.3e" % (npoints, nsq, worst))


main()
