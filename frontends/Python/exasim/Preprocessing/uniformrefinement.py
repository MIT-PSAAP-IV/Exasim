import numpy as np

from .createdgnodes import createdgnodes
from .facenumbering import facenumbering
from .mkshape import mkshape


def uniformrefinemesh(mesh, app, master, elem2cpu=None):
    """Uniformly refine a Python frontend mesh in parent-major order."""
    nlevel = int(app.get('uniformrefinementlevel', 0))
    if nlevel == 0:
        return mesh, elem2cpu
    if nlevel < 0 or nlevel != app.get('uniformrefinementlevel', 0):
        raise ValueError("uniformrefinementlevel must be a nonnegative integer.")
    if 'uhat' in mesh and len(mesh['uhat']) > 0:
        raise ValueError("uniformrefinementlevel > 0 cannot prolongate the face-based mesh['uhat'] field.")

    if _isempty(mesh.get('dgnodes', [])):
        try:
            f = facenumbering(mesh['p'], mesh['t'], app['elemtype'], mesh['boundaryexpr'], mesh['periodicexpr'])[0]
            mesh['dgnodes'] = createdgnodes(mesh['p'], mesh['t'], f, mesh['curvedboundary'],
                                            mesh['curvedboundaryexpr'], app['porder'])
        except Exception:
            pass

    T = _refinement_template(app['nd'], app['elemtype'], app['nve'])
    R = _refinement_prolongation(T, master['xpe'], app['porder'])

    ne0 = mesh['t'].shape[1]
    np0 = mesh['p'].shape[1]
    for name in ['dgnodes', 'udg', 'vdg', 'wdg']:
        _check_field(mesh, name, master['xpe'].shape[0], ne0)
    if elem2cpu is not None:
        elem2cpu = np.asarray(elem2cpu, dtype=int).reshape(-1, order='F')
        if elem2cpu.size == 0:
            elem2cpu = None
        elif elem2cpu.size != ne0:
            raise ValueError("uniformrefinement: elem2cpu must contain one entry per parent element.")

    for _ in range(nlevel):
        mesh['p'], mesh['t'] = _refine_connectivity(mesh['p'], mesh['t'], T, R, mesh.get('dgnodes', []))
        for name in ['dgnodes', 'udg', 'vdg', 'wdg']:
            _prolong_mesh_field(mesh, name, R['P'])
        if elem2cpu is not None:
            elem2cpu = np.repeat(elem2cpu, T['nchild'])

    print("uniformrefinementlevel = %d: ne %d -> %d, np %d -> %d" %
          (nlevel, ne0, mesh['t'].shape[1], np0, mesh['p'].shape[1]))
    return mesh, elem2cpu


def coarseelem2cpu(dmd, ne):
    elem2cpu = -np.ones(ne, dtype=int)
    for i, part in enumerate(dmd):
        owned = int(np.sum(np.asarray(part['elempartpts']).reshape(-1, order='F')[:2]))
        elem = np.asarray(part['elempart']).reshape(-1, order='F')[:owned]
        elem2cpu[elem] = i
    if np.any(elem2cpu < 0):
        raise ValueError("coarseelem2cpu: some coarse elements do not have an owner rank.")
    return elem2cpu


def _refinement_template(nd, elemtype, nve):
    xlat = []
    children = []

    def addpt(*x):
        xlat.append(tuple(x[:nd]))

    def addchild(v):
        children.append(tuple(v))

    if nd == 1 or elemtype == 1:
        ny = 3 if nd >= 2 else 1
        nz = 3 if nd == 3 else 1
        for k in range(nz):
            for j in range(ny):
                for i in range(3):
                    addpt(0.5*i, 0.5*j, 0.5*k)
        lat = lambda i, j, k: i + 3*(j + 3*k)
        if nd == 1:
            addchild([lat(0, 0, 0), lat(1, 0, 0)])
            addchild([lat(1, 0, 0), lat(2, 0, 0)])
        elif nd == 2:
            for cy in range(2):
                for cx in range(2):
                    addchild([lat(cx, cy, 0), lat(cx+1, cy, 0),
                              lat(cx+1, cy+1, 0), lat(cx, cy+1, 0)])
        else:
            for cz in range(2):
                for cy in range(2):
                    for cx in range(2):
                        addchild([lat(cx, cy, cz), lat(cx+1, cy, cz),
                                  lat(cx+1, cy+1, cz), lat(cx, cy+1, cz),
                                  lat(cx, cy, cz+1), lat(cx+1, cy, cz+1),
                                  lat(cx+1, cy+1, cz+1), lat(cx, cy+1, cz+1)])
    elif nd == 2:
        addpt(0, 0, 0); addpt(1, 0, 0); addpt(0, 1, 0)
        addpt(0.5, 0, 0); addpt(0.5, 0.5, 0); addpt(0, 0.5, 0)
        addchild([0, 3, 5])
        addchild([3, 1, 4])
        addchild([5, 4, 2])
        addchild([3, 4, 5])
    elif nd == 3:
        addpt(0, 0, 0); addpt(1, 0, 0); addpt(0, 1, 0); addpt(0, 0, 1)
        addpt(0.5, 0, 0); addpt(0, 0.5, 0); addpt(0, 0, 0.5)
        addpt(0.5, 0.5, 0); addpt(0.5, 0, 0.5); addpt(0, 0.5, 0.5)
        addchild([0, 4, 5, 6])
        addchild([4, 1, 7, 8])
        addchild([5, 7, 2, 9])
        addchild([6, 8, 9, 3])
        addchild([4, 5, 6, 8])
        addchild([4, 5, 7, 8])
        addchild([5, 6, 8, 9])
        addchild([5, 7, 8, 9])

    child = np.asarray(children, dtype=int).T
    if child.size != nve*child.shape[1]:
        raise ValueError("uniformrefinement: unsupported element.")
    xlat = np.asarray(xlat, dtype=float)
    phi = _linear_basis(xlat, nd, elemtype)
    nsup = np.zeros(xlat.shape[0], dtype=int)
    sup = np.zeros((nve, xlat.shape[0]), dtype=int)
    for l in range(xlat.shape[0]):
        s = np.where(phi[l, :] > 1e-12)[0]
        nsup[l] = s.size
        sup[:s.size, l] = s
    if elemtype == 0 and nd >= 2:
        for c in range(child.shape[1]):
            cv = child[:, c]
            E = xlat[cv[1:nd+1], :] - xlat[cv[0], :]
            if np.linalg.det(E[:, :nd]) < 0:
                child[[nd-1, nd], c] = child[[nd, nd-1], c]
    return {'nd': nd, 'elemtype': elemtype, 'nve': nve, 'xlat': xlat,
            'nlat': xlat.shape[0], 'nchild': child.shape[1],
            'child': child, 'nsup': nsup, 'sup': sup}


def _refinement_prolongation(T, xpe, porder):
    npe = xpe.shape[0]
    phielem = _linear_basis(xpe, T['nd'], T['elemtype'])
    P = np.zeros((npe, npe, T['nchild']))
    Slat = mkshape(porder, xpe, T['xlat'], T['elemtype'])[:, :, 0]
    for c in range(T['nchild']):
        pts = np.zeros((npe, T['nd']))
        for d in range(T['nd']):
            for v in range(T['nve']):
                pts[:, d] += phielem[:, v]*T['xlat'][T['child'][v, c], d]
        shap = mkshape(porder, xpe, pts, T['elemtype'])[:, :, 0]
        P[:, :, c] = shap.T
        if np.max(np.abs(np.sum(P[:, :, c], axis=1) - 1.0)) > 1e-8:
            raise ValueError("uniformrefinement: prolongation is not a partition of unity.")
    return {'P': P, 'Slat': Slat}


def _refine_connectivity(p, t, T, R, dgnodes):
    nd, npv = p.shape
    ne = t.shape[1]
    pref = np.array(p, copy=True, order='F')
    tref = np.zeros((T['nve'], ne*T['nchild']), dtype=t.dtype, order='F')
    index = {}
    latid = np.zeros(T['nlat'], dtype=int)
    has_dgnodes = not _isempty(dgnodes)
    for e in range(ne):
        te = t[:, e]
        for l in range(T['nlat']):
            ns = T['nsup'][l]
            if ns == 1:
                latid[l] = te[T['sup'][0, l]]
                continue
            support = tuple(sorted(int(te[v]) for v in T['sup'][:ns, l]))
            if support not in index:
                new_id = npv
                npv += 1
                index[support] = new_id
                if has_dgnodes:
                    x = np.array([np.dot(R['Slat'][:, l], dgnodes[:, d, e]) for d in range(nd)])
                else:
                    x = np.mean(p[:, list(support)], axis=1)
                pref = np.column_stack((pref, x))
            latid[l] = index[support]
        for c in range(T['nchild']):
            tref[:, e*T['nchild'] + c] = latid[T['child'][:, c]]
    return np.asarray(pref, order='F'), tref


def _prolong_mesh_field(mesh, name, P):
    if name not in mesh or _isempty(mesh[name]):
        return
    f = mesh[name]
    npe, nc, ne = f.shape
    nchild = P.shape[2]
    g = np.zeros((npe, nc, ne*nchild), dtype=f.dtype, order='F')
    for e in range(ne):
        for c in range(nchild):
            g[:, :, e*nchild + c] = P[:, :, c] @ f[:, :, e]
    mesh[name] = g


def _check_field(mesh, name, npe, ne):
    if name not in mesh or _isempty(mesh[name]):
        return
    if mesh[name].shape[0] != npe or mesh[name].shape[2] != ne:
        raise ValueError("uniformrefinement: mesh['%s'] must have size npe x ncomp x ne." % name)


def _isempty(a):
    return a is None or (isinstance(a, list) and len(a) == 0) or np.size(a) == 0


def _linear_basis(pts, nd, elemtype):
    if nd == 1:
        xi = pts[:, 0]
        return np.column_stack((1-xi, xi))
    if nd == 2 and elemtype == 0:
        xi = pts[:, 0]; eta = pts[:, 1]
        return np.column_stack((1-xi-eta, xi, eta))
    if nd == 2 and elemtype == 1:
        xi = pts[:, 0]; eta = pts[:, 1]
        return np.column_stack(((1-xi)*(1-eta), xi*(1-eta), xi*eta, (1-xi)*eta))
    if nd == 3 and elemtype == 0:
        xi = pts[:, 0]; eta = pts[:, 1]; zeta = pts[:, 2]
        return np.column_stack((1-xi-eta-zeta, xi, eta, zeta))
    if nd == 3 and elemtype == 1:
        xi = pts[:, 0]; eta = pts[:, 1]; zeta = pts[:, 2]
        return np.column_stack(((1-xi)*(1-eta)*(1-zeta), xi*(1-eta)*(1-zeta),
                                xi*eta*(1-zeta), (1-xi)*eta*(1-zeta),
                                (1-xi)*(1-eta)*zeta, xi*(1-eta)*zeta,
                                xi*eta*zeta, (1-xi)*eta*zeta))
    raise ValueError("uniformrefinement: unsupported element.")
