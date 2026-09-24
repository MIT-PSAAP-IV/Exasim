function props = local_interp_database(db, xi, e)
sz = size(xi);
xi = xi(:);
e = e(:);
props = zeros(numel(xi), double(db.database.nprop));

for k = 1:numel(xi)
    state = MaterialMeshEvaluate(db.material, [xi(k), e(k)]);
    for ip = 1:numel(db.propertyNames)
        props(k,ip) = state.(char(db.propertyNames(ip)));
    end
end

if numel(sz) > 2 || (numel(sz) == 2 && min(sz) > 1)
    props = reshape(props, [sz double(db.database.nprop)]);
end
end
