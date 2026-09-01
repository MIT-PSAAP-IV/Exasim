function database = read_material_database_bin(filename)
%READ_MATERIAL_DATABASE_BIN Read compact binary material database.

fid = fopen(filename, "r");
if fid < 0, error("Could not open %s for reading.", filename); end
cleanup = onCleanup(@() fclose(fid));
data = fread(fid, Inf, "double");
if numel(data) < 5
    error("material.bin database file is too short.");
end
nstate = checked_int(data(1));
nprop = checked_int(data(2));
dims = int64(arrayfun(@checked_int, data(3:5)));
nrows = prod(double(dims(1:double(nstate))));
ncols = double(nstate) + double(nprop);
expected = 5 + nrows*ncols;
if numel(data) ~= expected
    error("material.bin contains %d doubles, expected %d.", numel(data), expected);
end
rows = reshape(data(6:end), [ncols nrows])';
database = validate_material_database(struct("nstate", nstate, "nprop", nprop, "dims", dims, "rows", rows));
end

function v = checked_int(x)
if ~isfinite(x) || x ~= round(x)
    error("material database header entries must be integer-valued.");
end
v = int64(round(x));
end
