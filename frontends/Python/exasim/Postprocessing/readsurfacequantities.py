import os
import numpy


def _header(filename):
    """3-double header of an outbou*_np<r>.bin file, and the number of payload values."""
    with open(filename, "rb") as fid:
        hdr = numpy.fromfile(fid, dtype=numpy.float64, count=3)
    if hdr.size < 3:
        raise ValueError("readsurfacequantities: %s has no header" % filename)
    nvals = os.path.getsize(filename) // 8 - 3
    return [int(round(v)) for v in hdr], int(nvals)


def _payload(filename):
    return numpy.fromfile(filename, dtype=numpy.float64)[3:]


def _blocks(fileprefix, r):
    d = numpy.fromfile(fileprefix + "bouinfo_np" + str(r) + ".bin", dtype=numpy.float64)
    nblk = int(round(d[0]))
    return [(int(round(d[3 + 2 * j])), int(round(d[4 + 2 * j]))) for j in range(nblk)]


def readsurfacequantities(fileprefix, nranks=1, saveSolBouLoc=None):
    """Read the SurfaceQuantities written on the ibs boundaries (pde['saveSolBouFreq'] > 0).

    Parameters
    ----------
    fileprefix : str
        Output path prefix, e.g. os.path.join(dataoutpath, "out"); files are
        <fileprefix>bouinfo_np<r>.bin, <fileprefix>bousurf_np<r>.bin, ...
    nranks : int
        Number of MPI ranks (files np0 .. np<nranks-1>).
    saveSolBouLoc : None, 0 or 1
        Optional check of the run's evaluation location; the files themselves say which it is.

    Returns
    -------
    dict keyed by boundary id ib, each entry a dict with
        'values' : [np, nf, nsurfq, nsteps] surface quantities (np = npf face nodes when
                   saveSolBouLoc = 0, ngf face Gauss points when saveSolBouLoc = 1)
        'x'      : [np, nf, ncx, nsteps] coordinates of those points at each save
        'n'      : [np, nf, nd, nsteps]  unit normals at those points at each save
        'dA'     : [np, nf, nsteps] Gauss weight * face jacobian (saveSolBouLoc = 1 only, else
                   None), so (values[:, :, i, t] * dA[:, :, t]).sum() integrates quantity i
        'saveSolBouLoc' : 0 or 1
    Faces are concatenated over ranks (rank order) and face blocks with the same ib. Geometry
    is saved with every record, so it follows a moving or adapted mesh.
    """
    # Pass 1: layout -- blocks per rank, faces per boundary, and the common dimensions.
    ranks = []
    nftot = {}
    np_ = nsq = ncx = nd = nsteps = loc = None
    for r in range(nranks):
        blocks = _blocks(fileprefix, r)
        if sum(nf for _, nf in blocks) == 0:
            ranks.append((r, []))
            continue  # this rank owns no face on the ibs boundaries
        hs, ns_vals = _header(fileprefix + "bousurf_np" + str(r) + ".bin")
        hg, ng_vals = _header(fileprefix + "bousurfgeo_np" + str(r) + ".bin")
        hx, _ = _header(fileprefix + "bouxdg_np" + str(r) + ".bin")
        hn, _ = _header(fileprefix + "boundg_np" + str(r) + ".bin")
        nfbou = sum(nf for _, nf in blocks)
        if hs[1] != nfbou or hg[1] != nfbou or hg[0] != hs[0]:
            raise ValueError("readsurfacequantities: bousurf/bousurfgeo/bouinfo disagree on rank %d" % r)
        rloc = hg[2] - hx[2] - hn[2]          # 0: [x, n] (nodes), 1: [x, n, dA] (Gauss points)
        if rloc not in (0, 1):
            raise ValueError("readsurfacequantities: unexpected bousurfgeo width %d on rank %d" % (hg[2], r))
        reclen = hs[0] * nfbou * hs[2]
        rsteps = ns_vals // reclen if reclen > 0 else 0
        if rsteps * hs[0] * nfbou * hg[2] != ng_vals:
            raise ValueError("readsurfacequantities: bousurfgeo does not hold one record per save on rank %d" % r)
        dims = (hs[0], hs[2], hx[2], hn[2], rsteps, rloc)
        if np_ is None:
            np_, nsq, ncx, nd, nsteps, loc = dims
        elif dims != (np_, nsq, ncx, nd, nsteps, loc):
            raise ValueError("readsurfacequantities: ranks disagree on the output layout (rank %d)" % r)
        for ib, nf in blocks:
            nftot[ib] = nftot.get(ib, 0) + nf
        ranks.append((r, blocks))
    if np_ is None:
        return {}
    if saveSolBouLoc is not None and int(saveSolBouLoc) != loc:
        raise ValueError("readsurfacequantities: files were written with saveSolBouLoc = %d" % loc)

    # Allocate every result once.
    out = {}
    for ib, nf in nftot.items():
        out[ib] = {
            'values': numpy.zeros((np_, nf, nsq, nsteps)),
            'x': numpy.zeros((np_, nf, ncx, nsteps)),
            'n': numpy.zeros((np_, nf, nd, nsteps)),
            'dA': numpy.zeros((np_, nf, nsteps)) if loc == 1 else None,
            'saveSolBouLoc': loc,
        }
    ngeo = ncx + nd + loc

    # Pass 2: fill by face offset per boundary.
    offset = {ib: 0 for ib in nftot}
    for r, blocks in ranks:
        if not blocks:
            continue
        nfbou = sum(nf for _, nf in blocks)
        sdata = _payload(fileprefix + "bousurf_np" + str(r) + ".bin")
        gdata = _payload(fileprefix + "bousurfgeo_np" + str(r) + ".bin")
        srec = np_ * nfbou * nsq
        grec = np_ * nfbou * ngeo
        sk = gk = 0                            # offsets of this rank's blocks inside a record
        for ib, nf in blocks:
            o = offset[ib]
            e = out[ib]
            m = np_ * nf
            for t in range(nsteps):
                s0 = t * srec + sk
                e['values'][:, o:o + nf, :, t] = numpy.reshape(sdata[s0:s0 + m * nsq], (np_, nf, nsq), order='F')
                g0 = t * grec + gk
                e['x'][:, o:o + nf, :, t] = numpy.reshape(gdata[g0:g0 + m * ncx], (np_, nf, ncx), order='F')
                g0 += m * ncx
                e['n'][:, o:o + nf, :, t] = numpy.reshape(gdata[g0:g0 + m * nd], (np_, nf, nd), order='F')
                if loc == 1:
                    g0 += m * nd
                    e['dA'][:, o:o + nf, t] = numpy.reshape(gdata[g0:g0 + m], (np_, nf), order='F')
            sk += m * nsq
            gk += m * ngeo
            offset[ib] = o + nf
    return out
