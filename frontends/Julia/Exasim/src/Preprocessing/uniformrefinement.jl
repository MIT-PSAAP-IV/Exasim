function uniformrefinemesh(mesh, app, master, elem2cpu=Int[])

nlevel = app.uniformrefinementlevel;
if nlevel == 0
    return mesh, elem2cpu;
end
if nlevel < 0
    error("uniformrefinementlevel must be a nonnegative integer.");
end

if size(mesh.dgnodes, 3) == 0
    try
        f,~,~ = facenumbering(mesh.p, mesh.t, app.elemtype, mesh.boundaryexpr, mesh.periodicexpr);
        mesh.dgnodes = createdgnodes(mesh.p, mesh.t, f, mesh.curvedboundary, mesh.curvedboundaryexpr, app.porder);
    catch
    end
end

T = refinementtemplate(app.nd, app.elemtype, size(mesh.t, 1));
R = refinementprolongation(T, master.xpe, app.porder);

ne0 = size(mesh.t, 2);
np0 = size(mesh.p, 2);
checkfield(mesh.dgnodes, "dgnodes", size(master.xpe, 1), ne0);
checkfield(mesh.udg, "udg", size(master.xpe, 1), ne0);
checkfield(mesh.odg, "odg", size(master.xpe, 1), ne0);
checkfield(mesh.wdg, "wdg", size(master.xpe, 1), ne0);
if !isempty(elem2cpu) && length(elem2cpu) != ne0
    error("uniformrefinement: elem2cpu must contain one entry per parent element.");
end

for level = 1:nlevel
    xdg = size(mesh.dgnodes, 3) > 0 ? mesh.dgnodes : zeros(Float64, 0, 0, 0);
    mesh.p, mesh.t = refineconnectivity(mesh.p, mesh.t, T, R, xdg);
    mesh.dgnodes = prolongfield(mesh.dgnodes, R.P);
    mesh.udg = prolongfield(mesh.udg, R.P);
    mesh.odg = prolongfield(mesh.odg, R.P);
    mesh.wdg = prolongfield(mesh.wdg, R.P);
    if !isempty(elem2cpu)
        elem2cpu = reshape(repeat(reshape(elem2cpu, 1, :), T.nchild, 1), :);
    end
end

println("uniformrefinementlevel = $nlevel: ne $ne0 -> $(size(mesh.t, 2)), np $np0 -> $(size(mesh.p, 2))");
return mesh, elem2cpu;

end

function coarseelem2cpu(dmd, ne)

elem2cpu = -ones(Int, ne);
for i = 1:length(dmd)
    owned = sum(dmd[i].elempartpts[1:min(2, length(dmd[i].elempartpts))]);
    elem = dmd[i].elempart[1:owned];
    elem2cpu[elem] .= i - 1;
end
if any(elem2cpu .< 0)
    error("coarseelem2cpu: some coarse elements do not have an owner rank.");
end
return elem2cpu;

end

function refinementtemplate(nd, elemtype, nve)

xlat = Array{Float64,2}(undef, 0, nd);
children = Vector{Vector{Int}}();
addpt(x...) = (xlat = [xlat; reshape(Float64[x[i] for i=1:nd], 1, nd)]);
function addchild(v)
    push!(children, Int.(v));
end

if nd == 1 || elemtype == 1
    ny = nd >= 2 ? 3 : 1;
    nz = nd == 3 ? 3 : 1;
    for k = 0:(nz-1), j = 0:(ny-1), i = 0:2
        addpt(0.5*i, 0.5*j, 0.5*k);
    end
    lat(i,j,k) = 1 + i + 3*(j + 3*k);
    if nd == 1
        addchild([lat(0,0,0), lat(1,0,0)]);
        addchild([lat(1,0,0), lat(2,0,0)]);
    elseif nd == 2
        for cy = 0:1, cx = 0:1
            addchild([lat(cx,cy,0), lat(cx+1,cy,0), lat(cx+1,cy+1,0), lat(cx,cy+1,0)]);
        end
    else
        for cz = 0:1, cy = 0:1, cx = 0:1
            addchild([lat(cx,cy,cz), lat(cx+1,cy,cz), lat(cx+1,cy+1,cz), lat(cx,cy+1,cz),
                      lat(cx,cy,cz+1), lat(cx+1,cy,cz+1), lat(cx+1,cy+1,cz+1), lat(cx,cy+1,cz+1)]);
        end
    end
elseif nd == 2
    addpt(0, 0, 0); addpt(1, 0, 0); addpt(0, 1, 0);
    addpt(0.5, 0, 0); addpt(0.5, 0.5, 0); addpt(0, 0.5, 0);
    addchild([1, 4, 6]);
    addchild([4, 2, 5]);
    addchild([6, 5, 3]);
    addchild([4, 5, 6]);
elseif nd == 3
    addpt(0, 0, 0); addpt(1, 0, 0); addpt(0, 1, 0); addpt(0, 0, 1);
    addpt(0.5, 0, 0); addpt(0, 0.5, 0); addpt(0, 0, 0.5);
    addpt(0.5, 0.5, 0); addpt(0.5, 0, 0.5); addpt(0, 0.5, 0.5);
    addchild([1, 5, 6, 7]);
    addchild([5, 2, 8, 9]);
    addchild([6, 8, 3, 10]);
    addchild([7, 9, 10, 4]);
    addchild([5, 6, 7, 9]);
    addchild([5, 6, 8, 9]);
    addchild([6, 7, 9, 10]);
    addchild([6, 8, 9, 10]);
end

nchild = length(children);
if nchild == 0
    error("uniformrefinement: unsupported element.");
end
child = hcat(children...);
if length(child) != nve*nchild
    error("uniformrefinement: unsupported element.");
end

phi = linearbasis(xlat, nd, elemtype);
nlat = size(xlat, 1);
nsup = zeros(Int, nlat);
sup = zeros(Int, nve, nlat);
for l = 1:nlat
    s = findall(phi[l, :] .> 1e-12);
    nsup[l] = length(s);
    sup[1:length(s), l] = s;
end
if elemtype == 0 && nd >= 2
    for c = 1:nchild
        cv = child[:, c];
        E = xlat[cv[2:nd+1], :] .- xlat[cv[1], :]';
        if localdet(E[:, 1:nd]) < 0
            child[[nd, nd+1], c] = child[[nd+1, nd], c];
        end
    end
end
return (nd=nd, elemtype=elemtype, nve=nve, xlat=xlat, nlat=nlat,
        nchild=nchild, child=child, nsup=nsup, sup=sup);

end

function localdet(A)

if size(A, 1) == 2
    return A[1,1]*A[2,2] - A[1,2]*A[2,1];
elseif size(A, 1) == 3
    return A[1,1]*(A[2,2]*A[3,3] - A[2,3]*A[3,2]) -
           A[1,2]*(A[2,1]*A[3,3] - A[2,3]*A[3,1]) +
           A[1,3]*(A[2,1]*A[3,2] - A[2,2]*A[3,1]);
end
error("uniformrefinement: unsupported determinant size.");

end

function refinementprolongation(T, xpe, porder)

npe = size(xpe, 1);
phielem = linearbasis(xpe, T.nd, T.elemtype);
P = zeros(Float64, npe, npe, T.nchild);
Slat = mkshape(porder, xpe, T.xlat, T.elemtype)[:, :, 1];
for c = 1:T.nchild
    pts = zeros(Float64, npe, T.nd);
    for d = 1:T.nd, v = 1:T.nve
        pts[:, d] += phielem[:, v]*T.xlat[T.child[v, c], d];
    end
    shap = mkshape(porder, xpe, pts, T.elemtype)[:, :, 1];
    P[:, :, c] = shap';
    if maximum(abs.(sum(P[:, :, c], dims=2) .- 1.0)) > 1e-8
        error("uniformrefinement: prolongation is not a partition of unity.");
    end
end
return (P=P, Slat=Slat);

end

function refineconnectivity(p, t, T, R, dgnodes)

nd,np = size(p);
ne = size(t, 2);
pref = copy(p);
tref = zeros(Int, T.nve, ne*T.nchild);
index = Dict{String,Int}();
latid = zeros(Int, T.nlat);
hasdgnodes = size(dgnodes, 3) > 0;

for e = 1:ne
    te = t[:, e];
    for l = 1:T.nlat
        ns = T.nsup[l];
        if ns == 1
            latid[l] = te[T.sup[1, l]];
            continue;
        end
        support = sort(te[T.sup[1:ns, l]]);
        key = join(support, "_");
        if haskey(index, key)
            latid[l] = index[key];
        else
            id = np + 1;
            np = id;
            index[key] = id;
            if hasdgnodes
                x = [sum(R.Slat[:, l].*dgnodes[:, d, e]) for d=1:nd];
            else
                x = vec(mean(p[:, support], dims=2));
            end
            pref = [pref x];
            latid[l] = id;
        end
    end
    for c = 1:T.nchild
        tref[:, (e-1)*T.nchild + c] = latid[T.child[:, c]];
    end
end
return pref, tref;

end

function prolongfield(f, P)

if size(f, 3) == 0
    return f;
end
npe,nc,ne = size(f);
nchild = size(P, 3);
g = zeros(eltype(f), npe, nc, ne*nchild);
for e = 1:ne, c = 1:nchild
    g[:, :, (e-1)*nchild + c] = P[:, :, c]*f[:, :, e];
end
return g;

end

function checkfield(f, name, npe, ne)

if size(f, 3) == 0
    return;
end
if size(f, 1) != npe || size(f, 3) != ne
    error("uniformrefinement: mesh.$name must have size npe x ncomp x ne.");
end

end

function linearbasis(pts, nd, elemtype)

if nd == 1
    xi = pts[:, 1];
    return [1.0 .- xi xi];
elseif nd == 2 && elemtype == 0
    xi = pts[:, 1]; eta = pts[:, 2];
    return [1.0 .- xi .- eta xi eta];
elseif nd == 2 && elemtype == 1
    xi = pts[:, 1]; eta = pts[:, 2];
    return [(1.0 .- xi).*(1.0 .- eta) xi.*(1.0 .- eta) xi.*eta (1.0 .- xi).*eta];
elseif nd == 3 && elemtype == 0
    xi = pts[:, 1]; eta = pts[:, 2]; zeta = pts[:, 3];
    return [1.0 .- xi .- eta .- zeta xi eta zeta];
elseif nd == 3 && elemtype == 1
    xi = pts[:, 1]; eta = pts[:, 2]; zeta = pts[:, 3];
    return [(1.0 .- xi).*(1.0 .- eta).*(1.0 .- zeta) xi.*(1.0 .- eta).*(1.0 .- zeta) xi.*eta.*(1.0 .- zeta) (1.0 .- xi).*eta.*(1.0 .- zeta) (1.0 .- xi).*(1.0 .- eta).*zeta xi.*(1.0 .- eta).*zeta xi.*eta.*zeta (1.0 .- xi).*eta.*zeta];
end
error("uniformrefinement: unsupported element.");

end
