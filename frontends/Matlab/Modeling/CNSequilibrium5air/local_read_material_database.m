function db = local_read_material_database(filename)
% Read the text material database and construct the high-order material
% interpolation mesh used by frontends/Matlab/Materials.  The equilibrium-air
% tables are generated on dimensions compatible with porder=5.
porder = 5;
database = read_material_dat(filename);
if double(database.nstate) ~= 2 || double(database.nprop) < 15
    error('Unexpected equilibrium-air database format in %s.', filename);
end

stateNames = ["xi", "e"];
propertyNames = ["p", "T", "mu", "kappa_equi", "kappa_chem", "a_equi", ...
                 "p_xi", "p_e", "T_xi", "T_e", ...
                 "Y_N2", "Y_O2", "Y_NO", "Y_N", "Y_O"];
if double(database.nprop) > numel(propertyNames)
    propertyNames = [propertyNames, "property" + string((numel(propertyNames)+1):double(database.nprop))];
end
material = material_database_to_mesh(database, porder, stateNames, propertyNames);

xi = unique(database.rows(:,1));
e = unique(database.rows(:,2));
db = struct();
db.header = double([database.nstate, database.nprop, database.dims]);
db.xi = xi;
db.e = e;
db.database = database;
db.material = material;
db.porder = porder;
db.propertyNames = propertyNames;
end
