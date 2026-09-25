import os
import numpy


def _readbou(filename):
    """Read one outbou*_np<r>.bin file: (3-double header, flat float64 payload)."""
    tm = numpy.fromfile(filename, dtype=numpy.float64)
    if tm.size < 3:
        raise ValueError("readsurfacequantities: %s has no header" % filename)
    return numpy.int_(numpy.round(tm[0:3])), tm[3:]


def _splitblocks(data, blocks, npts, ncomp):
    """Split a per-block concatenation into {ib: [npts, nf_b, ncomp] arrays} (in file order)."""
    out = []
    k = 0
    for ib, nf in blocks:
        n = npts * nf * ncomp
        out.append((ib, numpy.reshape(data[k:k + n], (npts, nf, ncomp), order='F')))
        k += n
    return out, k


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
        Evaluation location used by the run. None (default) infers it: 1 when a
        bousurfgeo file exists (or the point count differs from npf), else 0. Pass it
        explicitly if the output directory may hold stale files from another run.

    Returns
    -------
    dict keyed by boundary id ib, each entry a dict with
        'values' : [np, nf, nsurfq, nsteps] surface quantities (np = npf face nodes when
                   saveSolBouLoc = 0, ngf face Gauss points when saveSolBouLoc = 1)
        'x'      : [np, nf, ncx] coordinates of those points
        'n'      : [np, nf, nd]  unit normals at those points
        'dA'     : [np, nf] quadrature weight * face jacobian (saveSolBouLoc = 1 only, else None),
                   so (values[:, :, i, t] * dA).sum() integrates quantity i over the boundary
        'saveSolBouLoc' : 0 or 1
    Faces are concatenated across ranks (rank order) and face blocks with the same ib.
    """
    parts = {}
    for r in range(nranks):
        hdr, info = _readbou(fileprefix + "bouinfo_np" + str(r) + ".bin")
        nblk = hdr[0]
        blocks = [(int(round(info[2 * j])), int(round(info[2 * j + 1]))) for j in range(nblk)]
        if nblk == 0:
            continue  # this rank owns no face on the ibs boundaries
        nfbou = sum(nf for _, nf in blocks)

        hs, sdata = _readbou(fileprefix + "bousurf_np" + str(r) + ".bin")
        npts, nfs, nsurfq = hs
        if nfs != nfbou:
            raise ValueError("readsurfacequantities: bousurf/bouinfo face counts differ on rank %d" % r)
        reclen = npts * nfbou * nsurfq
        nsteps = sdata.size // reclen if reclen > 0 else 0
        recs = [_splitblocks(sdata[t * reclen:(t + 1) * reclen], blocks, npts, nsurfq)[0]
                for t in range(nsteps)]

        hx, xdata = _readbou(fileprefix + "bouxdg_np" + str(r) + ".bin")
        hn, ndata = _readbou(fileprefix + "boundg_np" + str(r) + ".bin")
        npf, ncx, nd = hx[0], hx[2], hn[2]
        geofile = fileprefix + "bousurfgeo_np" + str(r) + ".bin"
        if saveSolBouLoc is None:
            loc = 1 if (npts != npf or os.path.isfile(geofile)) else 0
        else:
            loc = int(saveSolBouLoc)
        if loc == 1:
            # saveSolBouLoc = 1: Gauss-point geometry, per block [xg | nlg | dA].
            hg, gdata = _readbou(geofile)
            ngf, ncx, nd = hg[0], hg[2] - 1 - hn[2], hn[2]
            if ngf != npts:
                raise ValueError("readsurfacequantities: bousurfgeo/bousurf point counts differ on rank %d" % r)
            geo = []
            k = 0
            for ib, nf in blocks:
                n = ngf * nf
                xg = numpy.reshape(gdata[k:k + n * ncx], (ngf, nf, ncx), order='F'); k += n * ncx
                ng = numpy.reshape(gdata[k:k + n * nd], (ngf, nf, nd), order='F'); k += n * nd
                dA = numpy.reshape(gdata[k:k + n], (ngf, nf), order='F'); k += n
                geo.append((ib, xg, ng, dA))
        else:
            xb = _splitblocks(xdata, blocks, npf, ncx)[0]
            nb = _splitblocks(ndata, blocks, npf, nd)[0]
            geo = [(xb[j][0], xb[j][1], nb[j][1], None) for j in range(len(blocks))]

        for j, (ib, nf) in enumerate(blocks):
            p = parts.setdefault(ib, {'values': [], 'x': [], 'n': [], 'dA': [], 'saveSolBouLoc': loc})
            s = numpy.zeros((npts, nf, nsurfq, nsteps))
            for t in range(nsteps):
                s[:, :, :, t] = recs[t][j][1]
            p['values'].append(s)
            p['x'].append(geo[j][1])
            p['n'].append(geo[j][2])
            p['dA'].append(geo[j][3])

    out = {}
    for ib, p in parts.items():
        out[ib] = {
            'values': numpy.concatenate(p['values'], axis=1),
            'x': numpy.concatenate(p['x'], axis=1),
            'n': numpy.concatenate(p['n'], axis=1),
            'dA': numpy.concatenate(p['dA'], axis=1) if p['saveSolBouLoc'] == 1 else None,
            'saveSolBouLoc': p['saveSolBouLoc'],
        }
    return out
