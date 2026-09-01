function write_material_dat(filename, database)
%WRITE_MATERIAL_DAT Write text material database.

database = validate_material_database(database);
fid = fopen(filename, "w");
if fid < 0, error("Could not open %s for writing.", filename); end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "%.17g %.17g %.17g %.17g %.17g\n", double([database.nstate database.nprop database.dims]));
fmt = [repmat('%.17g ', 1, size(database.rows,2)-1), '%.17g\n'];
fprintf(fid, fmt, database.rows');
end
