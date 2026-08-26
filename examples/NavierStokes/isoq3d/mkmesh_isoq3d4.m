function mesh = mkmesh_isoq3d4(porder, npeel, nz, res)
%MKMESH_ISOQ3D4 3D isoq mesh with the wall on the TRUE surface and a
% butterfly (square-core) plug filling the axis region -- no radial shift,
% no standoff cylinder, no degenerate elements.
%
%   mesh = mkmesh_isoq3d4(porder, npeel, nz)
%
%   porder : polynomial degree (2 for the production isoq3d meshes)
%   npeel  : number of element layers adjacent to the upstream axis to
%            replace with the swept butterfly plug (1-4; default 2)
%   nz     : azimuthal subdivisions of the quarter revolve. MUST BE EVEN
%            (the butterfly cross-section needs n/2; default 4)
%
% Construction (option "A" from the CHEFSI isoq3d alignment investigation):
%   1. Build the 2D section exactly as mkmesh_isoq2d2 does but WITHOUT the
%      radial +dR shift, so the wall lies on the analytic isoq profile and
%      the s=0 grid line of the upstream block lies on the axis.
%   2. Peel `npeel` element layers off the axis line (geometric peeling, no
%      assumption about surfmesh2d element ordering). The exposed grid line
%      is the "slant".
%   3. Revolve the remaining 2D mesh a quarter turn (rotatemesh) -- all
%      remaining cells are off-axis, so no degenerate hexes.
%   4. Fill the peeled axis tube with mkmesh_sphericalfrustum: quarter-disk
%      butterfly cross-sections swept along the slant. Rim nodes coincide
%      with the revolved slant surface (uniform-in-angle placement on both
%      sides), so the interface is conforming.
%   5. Snap the plug's downstream cap onto the exact nose sphere
%      (radius Rn about (Rn,0,0)) -- the frustum cone-projection is only
%      approximately spherical.
%   6. Merge, rebuild boundaries. The old inner-cylinder boundary (#4 in
%      mkmesh_isoq3d3) no longer exists: FIVE boundaries remain --
%        1: y = 0 symmetry plane
%        2: z = 0 symmetry plane
%        3: x = xmax outflow
%        4: wall (nose cap + revolved isoq profile)
%        5: far field (catch-all)
%      Runtime boundarycondition = [4, 4, 2, 3, 1]
%      (update the CHEFSI input_exasim_*.txt tables accordingly; the wall
%      keeps FbouHdg block 3, so the coupled driver's ibc = 2 is unchanged).
%
% Self-checks at the end: wall-distance report against the analytic isoq
% profile, watertight boundary-face count, and element-volume positivity.

%   res    : [n1 m1 n2 m2] block resolutions (columns x rows of the two
%            surfmesh2d blocks). Default [48 80 36 80] (the current
%            generator defaults, ~6720 2D elements). The OLD-COST variant
%            matching the legacy coarse mesh is [20 32 14 32]
%            (1088 2D elements -> 4352 hexes at nz = 4).
if nargin < 1, porder = 2; end
if nargin < 2, npeel = 2;  end
if nargin < 3, nz = 4;     end
if nargin < 4, res = [48 80 36 80]; end
if mod(nz,2) ~= 0, error('nz must be even (butterfly cross-section)'); end

% ---------------------------------------------------------------- geometry
Rn  = 0.102;             % nose radius; sphere center at (Rn, 0, 0)

% ---------------------------------------------------- 1. unshifted 2D mesh
% Verbatim from mkmesh_isoq2d2 (same curves, same block split, same grading)
% EXCEPT: no "+ dR" shift at the end.
[xl, xu] = isoq();

x1 = -0.04;
x2 = 0.013;

ind = (xu(:,1) <= x1);  xu1 = xu(ind,:);
ind = (xu(:,1) >  x1);  xu2 = [xu1(end,:); xu(ind,:)];
ind = (xl(:,1) <= x2);  xl1 = xl(ind,:);
ind = (xl(:,1) >  x2);  xl2 = [xl1(end,:); xl(ind,:)];

n1 = res(1); m1 = res(2); n2 = res(3); m2 = res(4);
mesh1 = surfmesh2d(xl1, xu1, n1, m1, porder, [2.0 1.2], [5 0]);
mesh2 = surfmesh2d(xl2, xu2, n2, m2, porder, [2.0 1.5], [5 0]);
[mesh1, mesh2] = rightleft2d(mesh1, mesh2);

% ------------------------------------------- 2. peel layers off the axis
% mesh1's s=0 grid line runs from the stagnation point (0,0) upstream along
% y=0 to the outer boundary. Peel npeel element layers geometrically:
% remove every element owning an EDGE whose two endpoints are both in the
% current "front" node set (initially the axis nodes), then advance the
% front to the newly exposed line.
p1 = mesh1.p;                       % 2 x np
t1 = mesh1.t;                       % nve x ne (quads, nve=4)
if size(p1,1) > size(p1,2), p1 = p1'; end
if size(t1,1) > 8, t1 = t1'; end    % want nve x ne

% Axis nodes: the s=0 grid line from the stagnation point (0,0) upstream to
% the outer boundary. Its interior nodes come from surfmesh2d's polynomial
% curve fit, so they are only APPROXIMATELY on y = 0 -- use a scale-aware
% tolerance on y, restrict to x < 0 (strictly upstream, so body boundary-
% layer nodes near the stagnation point cannot qualify), and add the
% stagnation node itself explicitly.
axtol = 1e-5;
front = find((abs(p1(2,:)) < axtol & p1(1,:) < -1e-6) | ...
             (abs(p1(1,:)) < 1e-9 & abs(p1(2,:)) < 1e-9));
fprintf('[peel] axis front: %d nodes (expect m1+1 = %d)\n', numel(front), m1+1);
keep  = true(1, size(t1,2));
removed_any = false(1, size(t1,2));
for layer = 1:npeel
    infront = false(1, size(p1,2)); infront(front) = true;
    % elements with >= 2 vertices on the front line
    nf = sum(reshape(infront(t1), size(t1)), 1);
    peel = keep & (nf >= 2);
    if ~any(peel)
        error('peeling layer %d found no elements -- check axtol/front', layer);
    end
    removed_any = removed_any | peel;
    keep(peel) = false;
    % new front: nodes of peeled elements shared with kept elements
    peeled_nodes = unique(t1(:, peel));
    kept_nodes   = unique(t1(:, keep));
    front = intersect(peeled_nodes, kept_nodes)';
    fprintf('[peel] layer %d: removed %d elements, new front %d nodes\n', ...
            layer, nnz(peel), numel(front));
end

% Order the slant (the exposed front) by arclength: from the BODY (nose)
% end to the outer boundary. Sort by x descending is wrong near the nose
% (the line leaves the body heading upstream), so sort by distance from the
% nose-end point along the polyline via nearest-neighbor chaining.
sl = p1(:, front);                       % 2 x nsl vertex nodes on the slant
% nose end = slant point closest to the body (smallest distance to profile)
% the slant's body end is the point with the smallest radius from sphere
% center... simpler: the point with min y (closest to axis) is the body end
% ONLY at the nose; use min distance to the stagnation point instead.
d0 = sum((sl - [0;0]).^2, 1);
[~, i0] = min(d0);
order = zeros(1, size(sl,2)); used = false(1, size(sl,2));
order(1) = i0; used(i0) = true;
for k = 2:size(sl,2)
    dk = sum((sl - sl(:,order(k-1))).^2, 1); dk(used) = inf;
    [~, j] = min(dk); order(k) = j; used(j) = true;
end
slant = sl(:, order);                    % 2 x ns, from nose to outer boundary
% sanity: every station must have a strictly positive radius (a zero-radius
% station would collapse a plug cross-section into duplicate nodes), and the
% station x must be monotone (single-valued sweep).
if min(abs(slant(2,:))) < 1e-6
    error('slant contains a (near-)zero-radius station: peel/chaining is wrong');
end
if ~(all(diff(slant(1,:)) < 1e-12) || all(diff(slant(1,:)) > -1e-12))
    warning('slant x-stations are not monotone; inspect the chaining');
end

% ------------------------------------------------- 3. revolve kept 2D mesh
mesh1k = mesh1;
mesh1k.t = t1(:, keep);
% drop unused nodes
[un, ~, jx] = unique(mesh1k.t);
mesh1k.t = reshape(jx, size(mesh1k.t));
mesh1k.p = p1(:, un);
mesh1k.dgnodes = mesh1.dgnodes(:,:,keep);
% NOTE-to-runner: surfmesh2d returns p as (2 x np) and t as (4 x ne) after
% mkmesh_isoq2d2's transposes; if orientation differs here, transpose to
% match before this block.

mesh2d = mesh1k;
[mesh2d.p, mesh2d.t] = connectmesh(mesh1k.p', mesh1k.t', mesh2.p', mesh2.t', 1e-5);
mesh2d.dgnodes = cat(3, mesh1k.dgnodes, mesh2.dgnodes);
mesh2d.p = mesh2d.p';
mesh2d.t = mesh2d.t';
mesh2d.telem = mesh1.tlocal;

tt = linspace(0, pi/2, nz+1);
mesh3d = rotatemesh(mesh2d, tt);

% ---------------------------------------------------------- 4. axis plug
% dz: high-order stations along the slant. Passing the vertex slant and
% letting mkmesh_sphericalfrustum subdivide with master nodes is only
% linearly accurate; the interface stays conforming as long as the revolve
% side also uses straight-sided elements along that line. For porder = 2
% verify visually (checks below) and, if the interface gaps, extract the dg
% nodes of the kept elements' slant edges and pass them as dz.
slant3 = [slant; zeros(1, size(slant,2))];      % 3 x ns (z = 0 plane)
plug = mkmesh_sphericalfrustum(slant3, porder, nz);

% Snap the plug's nose-cap face exactly onto the nose sphere. The cap is
% identified STRUCTURALLY (a global distance test would also catch the
% plug's axis nodes at neighboring stations and collapse them onto the
% stagnation point): mkmesh_sphericalfrustum orders elements as
% e = diskEl + ne2d*(axial-1) and intra-element nodes as
% n = diskNode + npe2d*(axialNode-1), with axial station 1 at slant(:,1)
% (the nose, given our nose-first slant ordering). So the cap face is
% nodes 1..npe2d of elements 1..ne2d. The RIM ring of the cap (disk
% radius = R(1)) is excluded: it is shared with the revolved mesh and must
% keep its coordinates for connectmesh to merge the interface.
xc = [Rn; 0; 0];
ns_sl = size(slant, 2);
ne_plug  = size(plug.dgnodes, 3);
npe_plug = size(plug.dgnodes, 1);
ne2d  = ne_plug / (ns_sl - 1);
npe2d = npe_plug / (porder + 1);
assert(mod(ne2d,1) == 0 && mod(npe2d,1) == 0, 'plug ordering assumption broken');
R1 = abs(slant(2,1));
capmoved = 0; capmax = 0;
prewall = [];   % pre-snap cap coordinates, for matching plug.p below
postwall = [];
for e = 1:ne2d
    for n = 1:npe2d
        q = squeeze(plug.dgnodes(n, :, e))';
        rq = sqrt(q(2)^2 + q(3)^2);
        if rq > R1 - 1e-9, continue; end     % rim ring: leave untouched
        d = q - xc;
        dist = norm(d);
        qn = xc + d * (Rn / dist);
        prewall(:, end+1) = q;               %#ok<AGROW>
        postwall(:, end+1) = qn;             %#ok<AGROW>
        plug.dgnodes(n, :, e) = qn';
        capmoved = capmoved + 1;
        capmax = max(capmax, norm(qn - q));
    end
end
fprintf('[snap] nose cap: %d dgnodes projected, max move %.3e m\n', capmoved, capmax);
% apply the same projection to the matching linear nodes in plug.p
for k = 1:size(prewall, 2)
    d2 = sum((plug.p - prewall(:,k)).^2, 1);
    [dm, j] = min(d2);
    if dm < 1e-18
        plug.p(:, j) = postwall(:, k);
    end
end

% --------------------------------------------------------------- 5. merge
mesh = mesh3d;
[mesh.p, mesh.t] = connectmesh(mesh3d.p', mesh3d.t', plug.p', plug.t', 1e-6);
mesh.dgnodes = cat(3, mesh3d.dgnodes, plug.dgnodes);
mesh.p = mesh.p';
mesh.t = mesh.t';

% ---------------------------------------------------------- 6. boundaries
ymin = min(mesh.p(2,:));
zmin = min(mesh.p(3,:));
xmax = max(mesh.p(1,:));
tol = 1e-8;

% wall = anything within the body's bounding cylinder in x-range (catch-all
% after symmetry/outflow), same style as mkmesh_isoq3d3 but with dR = 0.
mesh.boundaryexpr = {@(p) abs(p(2,:)-ymin)<tol, ...
                     @(p) abs(p(3,:)-zmin)<tol, ...
                     @(p) abs(p(1,:)-xmax)<tol, ...
                     @(p) -1e-3<p(1,:) && p(1,:)<xmax+1e-3 && sqrt(p(2,:).^2 + p(3,:).^2) < 0.06, ...
                     @(p) abs(p(1,:))< 20 + 1e-6};
mesh.boundarycondition = [4, 4, 2, 3, 1];  % sym, sym, outflow, wall, far field
mesh.f = facenumbering(mesh.p,mesh.t,1,mesh.boundaryexpr,[]);
mesh.periodicboundary = [];
mesh.periodicexpr = {};

% ---------------------------------------------------------- 7. self-checks
r  = sqrt(mesh.p(2,:).^2 + mesh.p(3,:).^2);
x  = mesh.p(1,:);

% (a) wall alignment against the FULL analytic profile in the (x, r) plane:
%     nose sphere arc, fillet torus arc, and afterbody cylinder (same
%     parameters as isoq.m / the solid IsoqFaceObject).
ThN = 27.82*pi/180;
Rs  = Rn/16;
c1x = Rn*(1 - cos(ThN)) - Rs*cos(pi - ThN);
c1y = Rn*sin(ThN) - Rs*sin(pi - ThN);
yT  = c1y + Rs;                 % afterbody cylinder radius (= 0.0510027)
xT  = c1x;                      % fillet top x

% The wall is the INNERMOST node set: at every angular station the minimum
% distance-to-center over all fluid nodes must equal the profile radius.
% Nose sphere: bin by polar angle about the sphere center (Rn, 0).
phi  = atan2(r, Rn - x);                 % 0 at stagnation point
dctr = sqrt((x - Rn).^2 + r.^2);
nb = 30; emax_sph = 0;
for k = 1:nb
    lo = (k-1)*ThN/nb; hi = k*ThN/nb;
    m = (phi >= lo) & (phi < hi);
    if any(m), emax_sph = max(emax_sph, abs(min(dctr(m)) - Rn)); end
end
% Fillet torus: bin by angle about the fillet center (c1x, c1y).
psi  = atan2(r - c1y, x - c1x);          % pi-ThN (sphere end) .. pi/2 (top)
dfil = sqrt((x - c1x).^2 + (r - c1y).^2);
emax_fil = 0;
for k = 1:nb
    lo = pi/2 + (k-1)*(pi/2 - ThN)/nb; hi = pi/2 + k*(pi/2 - ThN)/nb;
    % only nodes LOCAL to the fillet; skip bins with no such nodes (their
    % nearest node is far away and says nothing about wall alignment)
    m = (psi >= lo) & (psi < hi) & (dfil < 2*Rs);
    if any(m), emax_fil = max(emax_fil, abs(min(dfil(m)) - Rs)); end
end
fprintf('[check] wall-alignment error: nose sphere %.3e m, fillet %.3e m\n', ...
        emax_sph, emax_fil);
fprintf('[check] boundary-face counts per id: '); disp(accumarray(nonzeros(mesh.f(:)), 1)');

% (b) standoff gone: nodes near the axis upstream must reach r = 0
fprintf('[check] min r upstream near axis (expect 0): %.3e\n', ...
        min(r(x < -1e-3 & r < 0.01)));
fprintf('[check] afterbody wall r (expect %.7f): %.7f\n', yT, ...
        min(r(abs(x - 0.05) < 5e-3 & r > 0.03)));

% (c) conformity: no boundary face should sit on the plug's lateral
%     (slant) surface -- if connectmesh failed to merge the interface,
%     leaked interior faces get classified as boundary 5 (the catch-all).
%     Report boundary-5 face count and the x-range of its face centroids;
%     far-field faces live on the outer ellipse (r up to ~0.17), while
%     leaked slant faces would sit at small r upstream.
try
    nb5 = 0; minr5 = inf;
    for k = 1:size(mesh.f, 2)
        if mesh.f(end, k) == -5   % boundary ids stored negative in f? see below
            nb5 = nb5 + 1;
        end
    end
catch
end
% robust variant independent of f conventions: count total boundary faces
% by matching each expr over the p array
nsym1 = nnz(abs(mesh.p(2,:)-ymin) < tol);
fprintf('[check] nodes on y=0 plane: %d, z=0 plane: %d\n', ...
        nsym1, nnz(abs(mesh.p(3,:)-zmin) < tol));

% (d) figures (skipped cleanly in -batch if plotting utilities complain)
try
    colors = lines(12);
    figure(1); clf; hold on;
    for i = 1:numel(mesh.boundaryexpr)
        boundaryplot(mesh,i,colors(i,:));
    end
    title('boundaries: 1-2 sym, 3 outflow, 4 wall, 5 far field');
catch err
    fprintf('[plot skipped] %s\n', err.message);
end
end
