function mesh = clemesh_airfoil(xf, yf, nxw, nflr, nflf, nfuf, nfur, nr, sps, spr, yref, lw, ll, porder, wakeopts)

if nargin < 15
  wakeopts = [];
end

[p,t] = clemeshparam6(nxw, nflr, nflf, nfuf, nfur, nr, sps, spr, yref);

if isempty(wakeopts) 
  [p, t, dgnodes] = clemeshmap(xf, yf, p, t, lw, ll, porder);
else
  [p, t, dgnodes] = clemeshmap2(xf, yf, p, t, lw, ll, porder, wakeopts);
end

mesh = clemesh(p, t, dgnodes, porder);
