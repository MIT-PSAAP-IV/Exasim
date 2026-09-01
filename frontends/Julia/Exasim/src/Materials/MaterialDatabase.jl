module Materials

export MaterialDatabase, read_material_dat, write_material_dat,
       read_material_database_bin, write_material_database_bin,
       validate_material_database, sort_material_database_rows

struct MaterialDatabase
    nstate::Int
    nprop::Int
    dims::NTuple{3,Int}
    rows::Matrix{Float64}

    function MaterialDatabase(nstate, nprop, dims, rows)
        ns = Int(nstate)
        np = Int(nprop)
        ds = Tuple(Int.(collect(dims)))
        length(ds) == 3 || error("material database requires n1,n2,n3")
        r = Array{Float64,2}(rows)
        _validate_material_database_shape(ns, np, ds, r)
        sorted = _sort_material_database_rows(ns, r)
        _validate_material_database_complete(ns, np, ds, sorted)
        return new(ns, np, ds, sorted)
    end
end

function read_material_dat(filename::AbstractString)
    lines = String[]
    for rawline in eachline(filename)
        line = replace(rawline, r"(#|%|//).*$" => "")
        line = strip(line)
        isempty(line) || push!(lines, line)
    end
    length(lines) >= 2 || error("material.dat must contain one header row and at least one sample row")
    header = parse.(Float64, split(lines[1]))
    ns, np, dims = _parse_material_database_header(header)
    ncols = ns + np
    rows = zeros(Float64, length(lines)-1, ncols)
    for (i, line) in enumerate(lines[2:end])
        vals = parse.(Float64, split(line))
        length(vals) == ncols || error("material.dat sample rows must contain $(ncols) numeric columns")
        rows[i,:] = vals
    end
    return MaterialDatabase(ns, np, dims, rows)
end

function write_material_dat(filename::AbstractString, database::MaterialDatabase)
    db = MaterialDatabase(database.nstate, database.nprop, database.dims, database.rows)
    open(filename, "w") do io
        println(io, join(Float64.([db.nstate, db.nprop, db.dims...]), " "))
        for i = 1:size(db.rows,1)
            println(io, join([string(x) for x in db.rows[i,:]], " "))
        end
    end
end

function read_material_database_bin(filename::AbstractString)
    data = reinterpret(Float64, read(filename))
    length(data) >= 5 || error("material.bin database file is too short")
    ns, np, dims = _parse_material_database_header(collect(data[1:5]))
    nrows = prod(dims[1:ns])
    ncols = ns + np
    expected = 5 + nrows*ncols
    length(data) == expected || error("material.bin contains $(length(data)) doubles, expected $(expected)")
    rows = collect(permutedims(reshape(collect(data[6:end]), ncols, nrows)))
    return MaterialDatabase(ns, np, dims, rows)
end

function write_material_database_bin(filename::AbstractString, database::MaterialDatabase)
    db = MaterialDatabase(database.nstate, database.nprop, database.dims, database.rows)
    open(filename, "w") do io
        write(io, Float64.([db.nstate, db.nprop, db.dims...]))
        write(io, vec(permutedims(db.rows)))
    end
end

validate_material_database(database::MaterialDatabase) = (
    _validate_material_database_shape(database.nstate, database.nprop, database.dims, database.rows);
    _validate_material_database_complete(database.nstate, database.nprop, database.dims, database.rows);
    nothing
)

sort_material_database_rows(database::MaterialDatabase) =
    _sort_material_database_rows(database.nstate, database.rows)

function _parse_material_database_header(header)
    length(header) == 5 || error("material database header must contain nstate nprop n1 n2 n3")
    all(isfinite, header) || error("material database header contains NaN or Inf")
    rounded = round.(header)
    all(header .== rounded) || error("material database header entries must be integer-valued")
    values = Int.(rounded)
    return values[1], values[2], Tuple(values[3:5])
end

function _validate_material_database_shape(ns::Int, np::Int, dims::NTuple{3,Int}, rows::Matrix{Float64})
    1 <= ns <= 3 || error("material database requires 1 <= nstate <= 3")
    np >= 1 || error("material database requires nprop >= 1")
    all(dims .> 0) || error("material database requires n1,n2,n3 > 0")
    ns == 1 && (dims[2] != 1 || dims[3] != 1) &&
        error("inactive dimensions for nstate=1 require n2=1 and n3=1")
    ns == 2 && dims[3] != 1 &&
        error("inactive dimension for nstate=2 requires n3=1")
    size(rows) == (prod(dims[1:ns]), ns+np) ||
        error("material database rows must have size n1*...*nstate by nstate+nprop")
    all(isfinite, rows) || error("material database contains NaN or Inf")
end

function _validate_material_database_complete(ns::Int, np::Int, dims::NTuple{3,Int}, rows::Matrix{Float64})
    axes = [_unique_sorted(rows[:,i]) for i=1:ns]
    Tuple(length.(axes)) == dims[1:ns] ||
        error("material database state coordinates do not match n1,n2,n3")
    keys = Set{Tuple}()
    for i = 1:size(rows,1)
        key = Tuple(rows[i,1:ns])
        key in keys && error("material database contains duplicated state points")
        push!(keys, key)
    end
    length(keys) == prod(length.(axes)) || error("material database is missing tensor-product state points")
end

function _sort_material_database_rows(ns::Int, rows::Matrix{Float64})
    order = sortperm(1:size(rows,1), by = i -> reverse(Tuple(rows[i,1:ns])))
    return rows[order,:]
end

_unique_sorted(v) = sort(unique(Float64.(v)))

end # module Materials
