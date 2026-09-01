function database = validate_material_database(database)
%VALIDATE_MATERIAL_DATABASE Validate provider-independent material database.

database.nstate = int64(database.nstate);
database.nprop = int64(database.nprop);
database.dims = int64(database.dims(:))';
database.rows = double(database.rows);
nstate = double(database.nstate);
nprop = double(database.nprop);
if nstate < 1 || nstate > 3
    error("material database requires 1 <= nstate <= 3.");
end
if nprop < 1
    error("material database requires nprop >= 1.");
end
if numel(database.dims) ~= 3 || any(database.dims <= 0)
    error("material database requires n1,n2,n3 > 0.");
end
if nstate == 1 && (database.dims(2) ~= 1 || database.dims(3) ~= 1)
    error("inactive dimensions for nstate=1 require n2=1 and n3=1.");
end
if nstate == 2 && database.dims(3) ~= 1
    error("inactive dimension for nstate=2 requires n3=1.");
end
expected_rows = prod(double(database.dims(1:nstate)));
expected_cols = nstate + nprop;
if ~isequal(size(database.rows), [expected_rows expected_cols])
    error("material database rows must have size n1*...*nstate by nstate+nprop.");
end
if any(~isfinite(database.rows(:)))
    error("material database contains NaN or Inf.");
end
database.rows = sort_material_database_rows(database);
states = database.rows(:,1:nstate);
if size(unique(states,"rows"),1) ~= size(states,1)
    error("material database contains duplicated state points.");
end
for i = 1:nstate
    if numel(unique(states(:,i))) ~= double(database.dims(i))
        error("material database state coordinates do not match n1,n2,n3.");
    end
end
end

function rows = sort_material_database_rows(database)
nstate = double(database.nstate);
keys = database.rows(:,1:nstate);
[~, order] = sortrows(keys, nstate:-1:1);
rows = database.rows(order,:);
end
