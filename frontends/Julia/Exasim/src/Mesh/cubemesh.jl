function cubemesh(m,n,o,elemtype)
# Generate mesh for unit cube

p = [[x; y; z] for x = 0.0:1.0/Float64(m):1.0, y = 0.0:1.0/Float64(n):1.0, z = 0.0:1.0/Float64(o):1.0][:]
p = hcat(p...);

m = m+1;
n = n+1;
o = o+1;

if elemtype==0
    t0=[[5 6 7 3]; [5 6 2 3]; [5 1 2 3]; [6 8 7 4]; [6 7 4 3]; [6 2 4 3]];

    map=[1 2];
    map=[map  map.+m];
    map=[map map.+m*n];
    t=map[t0];
    t=kron(t,ones(m-1,1))+kron(ones(size(t)),collect(0:m-2));
    t=kron(t,ones(n-1,1))+kron(ones(size(t)),collect(0:n-2).*m);
    t=kron(t,ones(o-1,1))+kron(ones(size(t)),collect(0:o-2).*(m*n));
else
    t = [1 2 m+2 m+1 m*n+1 m*n+2 m*n+m+2 m*n+m+1];
    t=kron(t,ones(o-1,1))+kron(ones(size(t)),collect(0:o-2).*(m*n));
    t=kron(t,ones(n-1,1))+kron(ones(size(t)),collect(0:n-2).*m);
    t=kron(t,ones(m-1,1))+kron(ones(size(t)),collect(0:m-2));
end

t = Int.(t');

if elemtype==0
    p, t = orient_tetrahedra(p, t);
end

return p, t

end

# Same result as the MATLAB cubemesh, which calls fixmesh: number the nodes in
# lexicographic (x, y, z) order and orient every tetrahedron positively (the
# solver rejects inverted elements).
function orient_tetrahedra(p, t)
    order = sortperm(collect(eachcol(p)), by = x -> (x[1], x[2], x[3]));
    rank = similar(order);
    rank[order] = 1:length(order);
    p = p[:, order];
    t = rank[t];
    for e = 1:size(t, 2)
        a = p[:, t[2, e]] - p[:, t[1, e]];
        b = p[:, t[3, e]] - p[:, t[1, e]];
        c = p[:, t[4, e]] - p[:, t[1, e]];
        volume = (a[2]*b[3] - a[3]*b[2])*c[1] + (a[3]*b[1] - a[1]*b[3])*c[2] +
                 (a[1]*b[2] - a[2]*b[1])*c[3];
        if volume < 0
            t[1, e], t[2, e] = t[2, e], t[1, e];
        end
    end
    return p, t
end
