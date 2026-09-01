function database = read_material_dat(filename)
%READ_MATERIAL_DAT Read text material database.

fid = fopen(filename, "r");
if fid < 0, error("Could not open %s for reading.", filename); end
cleanup = onCleanup(@() fclose(fid));
header = sscanf(fgetl(fid), "%f")';
if numel(header) ~= 5
    error("material.dat header must contain nstate nprop n1 n2 n3.");
end
nstate = checked_int(header(1));
nprop = checked_int(header(2));
dims = int64(arrayfun(@checked_int, header(3:5)));
rows = fscanf(fid, "%f", [nstate+nprop, Inf])';
database = validate_material_database(struct("nstate", nstate, "nprop", nprop, "dims", dims, "rows", rows));
end

function v = checked_int(x)
if ~isfinite(x) || x ~= round(x)
    error("material database header entries must be integer-valued.");
end
v = int64(round(x));
end
