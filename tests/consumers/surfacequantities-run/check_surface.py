#!/usr/bin/env python3
"""Checks for the surfacequantities-run consumer (stdlib only: no numpy on CI runners).

Builtin model 1 defines SurfaceQuantities = [u, f.n + tau*(u - uhat)] with f = kappa*q, i.e.
the second output is exactly its QoIboundary integrand. So:

  nodes <prefix> <np>          outbousurf at face nodes vs a recomputation from the
                               outbouudg / outboundg / outbouuhat files of the same run
  gauss <prefix> <np>          outbousurf at Gauss points: sum(s1*dA) == Boundary_QoI1
  union <prefix_a> <prefix_b> <np>
                               Boundary_QoI1 and the saved face count agree between the
                               ibs = [1, 2] run (a) and the single-id baseline run (b)

<prefix> is the run's output prefix, e.g. dataout/out.
"""
import struct
import sys

KAPPA = 1.0   # physicsparam[0]
TAU = 1.0     # tau[0]


def read_doubles(path):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack("<%dd" % (len(data) // 8), data))


def header_and_records(path):
    """(header, [record, ...]) for an outbou* file: 3-double header then fixed-size records."""
    d = read_doubles(path)
    hdr = [int(round(v)) for v in d[:3]]
    body = d[3:]
    n = hdr[0] * hdr[1] * hdr[2]
    if n == 0:
        return hdr, []
    if len(body) % n:
        raise SystemExit("FAIL: %s has %d values, not a multiple of %d" % (path, len(body), n))
    return hdr, [body[i:i + n] for i in range(0, len(body), n)]


def blocks(prefix, rank):
    d = read_doubles("%sbouinfo_np%d.bin" % (prefix, rank))
    nblk = int(round(d[0]))
    return [(int(round(d[3 + 2 * b])), int(round(d[4 + 2 * b]))) for b in range(nblk)]


def split(record, npts, blks, ncomp):
    """Per-block [npts*nf, ncomp] (point-major, component blocks) views of one record."""
    out, off = [], 0
    for _, nf in blks:
        n = npts * nf
        out.append([record[off + k * n: off + (k + 1) * n] for k in range(ncomp)])
        off += n * ncomp
    return out


def last_qoi(prefix):
    with open(prefix + "qoi.txt") as f:
        lines = [l.split() for l in f if l.strip()]
    head, last = lines[0], lines[-1]
    return {k: float(v) for k, v in zip(head, last)}


def check_nodes(prefix, nranks):
    worst, npoints, ibs = 0.0, 0, set()
    for r in range(nranks):
        blks = blocks(prefix, r)
        ibs.update(ib for ib, _ in blks)
        sh, srec = header_and_records("%sbousurf_np%d.bin" % (prefix, r))
        uh_, urec = header_and_records("%sbouudg_np%d.bin" % (prefix, r))
        hh, hrec = header_and_records("%sbouuhat_np%d.bin" % (prefix, r))
        nh, nrec = header_and_records("%sboundg_np%d.bin" % (prefix, r))
        npf, nfbou, nsq = sh
        if nsq != 2 or uh_[1] != nfbou or nfbou != sum(nf for _, nf in blks):
            raise SystemExit("FAIL: rank %d headers disagree: surf %s udg %s info %s" % (r, sh, uh_, blks))
        if not srec:
            continue
        if len(srec) != len(urec):
            raise SystemExit("FAIL: rank %d has %d surf records but %d udg records" % (r, len(srec), len(urec)))
        nc, nd = uh_[2], nh[2]
        S = split(srec[-1], npf, blks, nsq)
        U = split(urec[-1], npf, blks, nc)
        H = split(hrec[-1], npf, blks, hh[2])
        N = split(nrec[0], npf, blks, nd)
        for b in range(len(blks)):
            for i in range(len(S[b][0])):
                u, q1, q2 = U[b][0][i], U[b][1][i], U[b][2][i]
                fn = KAPPA * (q1 * N[b][0][i] + q2 * N[b][1][i]) + TAU * (u - H[b][0][i])
                worst = max(worst, abs(S[b][0][i] - u), abs(S[b][1][i] - fn))
                npoints += 1
    if npoints == 0:
        raise SystemExit("FAIL: no surface-quantity points written")
    if ibs != {1, 2}:
        raise SystemExit("FAIL: expected boundary blocks for ibs 1 and 2, got %s" % sorted(ibs))
    if worst > 1e-10:
        raise SystemExit("FAIL: nodal surface quantities differ from recomputation by %.3e" % worst)
    print("  [surfq nodes] %d points on ibs %s, max |diff| = %.2e" % (npoints, sorted(ibs), worst))


def check_gauss(prefix, nranks):
    total, npoints = 0.0, 0
    for r in range(nranks):
        blks = blocks(prefix, r)
        sh, srec = header_and_records("%sbousurf_np%d.bin" % (prefix, r))
        gh, grec = header_and_records("%sbousurfgeo_np%d.bin" % (prefix, r))
        ngf, nfbou, nsq = sh
        if gh[0] != ngf or gh[1] != nfbou or not srec:
            if nfbou == 0:
                continue
            raise SystemExit("FAIL: rank %d surf/geo headers disagree: %s %s" % (r, sh, gh))
        ncx_nd = gh[2] - 1
        S = split(srec[-1], ngf, blks, nsq)
        G = split(grec[0], ngf, blks, ncx_nd + 1)   # [x..., n..., dA]
        for b in range(len(blks)):
            dA = G[b][ncx_nd]
            total += sum(s * a for s, a in zip(S[b][1], dA))
            npoints += len(dA)
    qoi = last_qoi(prefix)["Boundary_QoI1"]
    rel = abs(total - qoi) / max(abs(qoi), 1e-300)
    if rel > 1e-9:
        raise SystemExit("FAIL: sum(s1*dA) = %.12e but Boundary_QoI1 = %.12e (rel %.2e)" % (total, qoi, rel))
    print("  [surfq gauss] %d points, sum(s1*dA) = %.10e == Boundary_QoI1 (rel %.1e)" % (npoints, total, rel))


def nfaces(prefix, nranks):
    return sum(nf for r in range(nranks) for _, nf in blocks(prefix, r))


def check_union(prefix_a, prefix_b, nranks):
    qa, qb = last_qoi(prefix_a), last_qoi(prefix_b)
    na, nb = nfaces(prefix_a, nranks), nfaces(prefix_b, nranks)
    for k in ("Domain_QoI1", "Boundary_QoI1"):
        rel = abs(qa[k] - qb[k]) / max(abs(qb[k]), 1e-300)
        if rel > 1e-9:
            raise SystemExit("FAIL: %s differs between ibs=[1,2] (%.12e) and ibs=1 (%.12e)" % (k, qa[k], qb[k]))
    if na != nb:
        raise SystemExit("FAIL: ibs=[1,2] saved %d boundary faces, ibs=1 saved %d" % (na, nb))
    print("  [surfq union] ibs=[1,2] == single id: %d faces, Boundary_QoI1 = %.10e" % (na, qa["Boundary_QoI1"]))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "nodes":
        check_nodes(sys.argv[2], int(sys.argv[3]))
    elif mode == "gauss":
        check_gauss(sys.argv[2], int(sys.argv[3]))
    elif mode == "union":
        check_union(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        raise SystemExit("unknown mode " + mode)
