from numpy import *

def cubemesh(m,n,o,elemtype):
# Generate mesh for unit cube

    p = [[x/m, y/n, z/o] for z in arange(0,o+1) for y in arange(0,n+1) for x in arange(0,m+1)];
    p = array(p,order='F').T;

    m = m+1;
    n = n+1;
    o = o+1;

    if elemtype==0:
        t0 = [[5, 6, 7, 3], [5, 6, 2, 3], [5, 1, 2, 3], [6, 8, 7, 4], [6, 7, 4, 3], [6, 2, 4, 3]];
        t0 = array(t0);

        map = array([1, 2]);
        map = concatenate([map,  map+m]);
        map = concatenate([map, map+m*n]);
        t=map[t0-1];
        t=kron(t,ones((m-1,1)))+kron(ones(t.shape),reshape(arange(0,m-1),(m-1,1)));
        t=kron(t,ones((n-1,1)))+kron(ones(t.shape),reshape(arange(0,n-1)*m,(n-1,1)));
        t=kron(t,ones((o-1,1)))+kron(ones(t.shape),reshape(arange(0,o-1)*(m*n),(o-1,1)));
    else:
        t = array([1, 2, m+2, m+1, m*n+1, m*n+2, m*n+m+2, m*n+m+1]);
        t = kron(t,ones((o-1,1)))+kron(ones(t.shape),reshape(arange(0,o-1)*(m*n),(o-1,1)));
        t = kron(t,ones((n-1,1)))+kron(ones(t.shape),reshape(arange(0,n-1)*m,(n-1,1)));
        t = kron(t,ones((m-1,1)))+kron(ones(t.shape),reshape(arange(0,m-1),(m-1,1)));

    t = t.transpose().astype(int)-1;

    if elemtype==0:
        p, t = orient_tetrahedra(p, t)

    return p, t


def orient_tetrahedra(p, t):
    # Same result as the MATLAB cubemesh, which calls fixmesh: number the nodes
    # in lexicographic (x, y, z) order and orient every tetrahedron positively
    # (the solver rejects inverted elements).
    order = lexsort((p[2], p[1], p[0]))
    rank = empty_like(order)
    rank[order] = arange(order.size)
    p = p[:, order]
    t = rank[t]
    x = p[:, t]
    volume = einsum('ij,ij->j', cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], axis=0),
                    x[:, 3] - x[:, 0])
    flip = volume < 0
    t[0, flip], t[1, flip] = t[1, flip], t[0, flip].copy()
    return p, t
