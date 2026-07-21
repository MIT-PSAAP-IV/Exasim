function [mesh, parts, info] = windturbine2d_single_airfoil_mesh(opts)
%WINDTURBINE2D_SINGLE_AIRFOIL_MESH Build one C-mesh airfoil component.

if nargin < 1 || isempty(opts)
    opts = windturbine2d_options();
end

[xf, yf] = read_foil(opts.airfoilFile);
xf = opts.airfoilScale * xf(:);
yf = opts.airfoilScale * yf(:);

c = opts.cmesh;
sps = [c.TEC, 1, 1, 1, 1, c.TEC, 1, 1, 1, 1, c.TEC];
[p, t] = clemeshparam6(c.nxw, c.nflr, c.nflf, c.nfuf, c.nfur, c.nr, ...
    sps, c.spr, c.yref);
[p, t, xdg, wakeinfo] = clemeshmap2(xf, yf, p, t, c.lw, c.ll, ...
    opts.porder, c.wakeopts);

[mesh, info] = windturbine2d_connect_cmesh_blocks(p, t, xdg, opts);
info.wake = wakeinfo;

parts.p = p;
parts.t = t;
parts.xdg = xdg;
parts.xf = xf;
parts.yf = yf;

end
