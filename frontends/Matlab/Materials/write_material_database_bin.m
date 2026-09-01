function write_material_database_bin(filename, database)
%WRITE_MATERIAL_DATABASE_BIN Write compact binary material database.

database = validate_material_database(database);
fid = fopen(filename, "w");
if fid < 0, error("Could not open %s for writing.", filename); end
cleanup = onCleanup(@() fclose(fid));
header = double([database.nstate database.nprop database.dims]);
fwrite(fid, header(:), "double");
fwrite(fid, database.rows', "double");
end
