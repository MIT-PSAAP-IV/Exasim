function mesh = clemesh_airfoil(xf, yf, nxw, nflr, nflf, nfuf, nfur, nr, sps, spr, yref, lw, ll, porder)

[p,t] = clemeshparam6(nxw, nflr, nflf, nfuf, nfur, nr, sps, spr, yref);
[p, t, dgnodes] = clemeshmap(xf, yf, p, t, lw, ll, porder);
mesh = clemesh(p, t, dgnodes, porder);
